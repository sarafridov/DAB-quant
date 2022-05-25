import os
# Fix the OpenSlide path for Windows
if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openslide')):
    os.add_dll_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.join('openslide', 'bin')))
import openslide
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from argparse import ArgumentParser
import glob
import time
import csv


parser = ArgumentParser()
parser.add_argument('--controls_folder', type=str, default=None, help='name of the folder where the control (unstained) image files are, \
    to be used for automating the choice of background_threshold and stain_threshold. If not provided, background_threshold and stain_threshold will be used.')
parser.add_argument('--folder', type=str, default='images', help='name of the folder where the image files to be processed are')
parser.add_argument('--reprocess_controls', action='store_true', default=False, help='re-process all control images (default is to only process new images)')
parser.add_argument('--reprocess', action='store_true', default=False, help='re-process all images (default is to only process new images)')
parser.add_argument('--rmb_precision', type=float, default=0.01, help='quantization precision for normalized red minus blue values')
parser.add_argument('--gray_precision', type=float, default=1, help='quantization precision for grayscale brightness')
parser.add_argument('--num_regions', type=int, default=10, help='number of image regions to sample')
parser.add_argument('--region_size', type=int, default=500, help='select regions with this many pixels on a side')
parser.add_argument('--include_full', action='store_true', default=False, help='include the full slide (slower), excludes whitespace')
parser.add_argument('--show_processed_images', action='store_true', default=False, help='save a copy of the normalized red - blue regions')
parser.add_argument('--stain_threshold', type=float, default=0.2, help='how red a pixel must be to be considered stained. If controls_folder is provided, stain_threshold is ignored.')
parser.add_argument('--include_background', action='store_true', default=False, help='include near-white pixels that are intermixed with cells')
parser.add_argument('--background_threshold', type=int, default=220, help='average brightness must be at least this value (out of 255) to be considered background. \
    If controls_folder is provided, background_threshold is ignored.')
parser.add_argument('--error_tolerance', type=float, default=0.001, help='what fraction of control (unstained) pixels to tolerate being misclassified as stained. \
    Used in conjunction with control (unstained) images to automatically choose stain_threshold.')
parser.add_argument('--background_fraction', type=float, default=0.5, help='what fraction of background (whitespace) to tolerate in a region')
parser.add_argument('--seed', type=int, default=0, help='random seed: defaults to 0; choose different numbers to randomize selected regions, or leave fixed for reproducibility')
parser.add_argument('--ext', type=str, default='.ndpi', help='file extension for scanned slides, must be readable by OpenSlide: https://openslide.org/formats/')

args = parser.parse_args()

np.random.seed(args.seed)

filenames = glob.glob(os.path.join(args.folder, '*' + args.ext))
plain_filenames = [name.split(os.path.sep)[-1] for name in filenames]

# Check if the exclude_regions.csv is already present. If not, create it.
exclude_dict = {}
for filename in plain_filenames:
    exclude_dict[filename] = []
if os.path.isfile(os.path.join(args.folder, 'exclude_regions.csv')):
    file = csv.DictReader(open(os.path.join(args.folder, 'exclude_regions.csv'), 'r'))
    for row in file:
        for filename in plain_filenames:
            if row[filename].isnumeric():
                exclude_dict[filename].append(int(row[filename]))
else:
    np.savetxt(os.path.join(args.folder, 'exclude_regions.csv'), [plain_filenames], fmt='%s', delimiter=',')

if args.controls_folder is not None:
    control_filenames = glob.glob(os.path.join(args.controls_folder, '*' + args.ext))
rmb_bins = np.linspace(-1, 254, num=int(256/args.rmb_precision), endpoint=True)  # Edges of the normalized red minus blue bins
gray_bins = np.linspace(0, 255, num=int(256/args.gray_precision), endpoint=True)  # Edges of the grayscale brightness bins
bins = np.arange(-3,3.05,0.01)  # Bins for rmb histogram plotting
output_filename = os.path.join(args.folder, 'stained_fractions.txt')
if args.reprocess:
    output_file = open(output_filename, 'w')  # Erase prior content if we are reprocessing
    output_file.close()

    
# Copied from skimage.filters
def _validate_image_histogram(image, hist, nbins=None):
    """Ensure that either image or hist were given, return valid histogram.
    If hist is given, image is ignored.
    Parameters
    ----------
    image : array or None
        Grayscale image.
    hist : array, 2-tuple of array, or None
        Histogram, either a 1D counts array, or an array of counts together
        with an array of bin centers.
    nbins : int, optional
        The number of bins with which to compute the histogram, if `hist` is
        None.
    Returns
    -------
    counts : 1D array of float
        Each element is the number of pixels falling in each intensity bin.
    bin_centers : 1D array
        Each element is the value corresponding to the center of each intensity bin.
    Raises
    ------
    ValueError : if image and hist are both None
    """
    if image is None and hist is None:
        raise Exception("Either image or hist must be provided.")

    if hist is not None:
        if isinstance(hist, (tuple, list)):
            counts, bin_centers = hist
        else:
            counts = hist
            bin_centers = np.arange(counts.size)
    else:
        counts, bin_centers = skimage.exposure.histogram(
                image.ravel(), nbins, source_range='image'
            )
    return counts.astype(float), bin_centers


# Copied from skimage.filters
def threshold_otsu(image=None, nbins=256, hist=None):
    if image is not None and image.ndim > 2 and image.shape[-1] in (3, 4):
        msg = "threshold_otsu is expected to work correctly only for " \
              "grayscale images; image shape {0} looks like an RGB image"
        warn(msg.format(image.shape))

    # Check if the image has more than one intensity value; if not, return that
    # value
    if image is not None:
        first_pixel = image.ravel()[0]
        if np.all(image == first_pixel):
            return first_pixel

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold


# From https://github.com/python/cpython/blob/3.10/Lib/bisect.py
def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
    return lo


def fill_grid(img, grid):
    (x_lim, y_lim, channels) = img.shape
    rmb = img.reshape(x_lim * y_lim, channels).astype(float)
    gray = np.mean(rmb, axis=1)  # Mean over color channels. Ranges from 0 to 255.
    rmb = (rmb[:,0] - rmb[:,2]) / np.maximum(1, rmb[:,2])  # Normalized red minus blue. Ranges from -1 to 255.
    # Sort by grayscale value
    idx = np.argsort(gray)
    gray = gray[idx]
    rmb = rmb[idx]
    oldindex = 0
    for i in range(len(gray_bins) - 1):
        index = bisect_left(gray, gray_bins[i+1])
        if i == len(gray_bins) - 2:
            sub_rmb = rmb[oldindex:-1]
        else:
            sub_rmb = rmb[oldindex:index]
        oldindex = index
        n, _ = np.histogram(sub_rmb, bins=rmb_bins)  # n is the number of pixels in each bin
        grid[i,:] += n
    return grid


def preprocess_control(filename):
    if os.path.exists(filename[0:-len(args.ext)] + '.csv') and not args.reprocess_controls:
        print(f'already processed {filename}')
        return
    print(f'processing {filename}')
    t1 = time.time()
    slide = openslide.OpenSlide(filename)
    dims = slide.dimensions
    # Partition the slide into grid cells
    nx = 16
    ny = 16
    xlen = dims[0] // 16
    ylen = dims[1] // 16
    # Iterate through the regions
    order = np.arange(nx*ny)
    x_coords = np.arange(0, dims[0], xlen)
    y_coords = np.arange(0, dims[1], ylen)
    grid = np.zeros((len(gray_bins) - 1, len(rmb_bins) - 1))
    for region in order:
        region_x = region // ny
        region_y = region - region_x * ny
        img = np.asarray(slide.read_region(location=(x_coords[region_x], y_coords[region_y]), level=0, size=(xlen, ylen)))
        grid = fill_grid(img, grid)
    np.savetxt(filename[0:-len(args.ext)] + '.csv', grid)
    print(f'preprocessing {filename} took {time.time() - t1} seconds')

    
def safe_otsu(hist):
    # Does Otsu thresholding, where the initial empty bins are first removed so they don't confuse Otsu
    start_idx = np.nonzero(hist)[0][0]
    thresh = threshold_otsu(hist=hist[start_idx:])
    return int(start_idx + thresh)


def red_minus_blue(img, include_background, background_threshold):
    (x_lim, y_lim, channels) = img.shape
    # Use the normalized difference between red and blue as a proxy for brown staining
    rmb = img.reshape(x_lim * y_lim, channels).astype(float)
    non_whitespace_idx = np.mean(rmb, axis=1) <= background_threshold
    rmb = (rmb[:,0] - rmb[:,2]) / np.maximum(1, rmb[:,2]) # Channel 0 is red, 2 is blue
    if not include_background:
        rmb = rmb[non_whitespace_idx]
    return rmb, 1 - np.sum(non_whitespace_idx)/len(non_whitespace_idx)


background_threshold = args.background_threshold
# Use control images to set background_threshold
if args.controls_folder is not None:
    t1 = time.time()
    grid = np.zeros((len(gray_bins) - 1, len(rmb_bins) - 1))
    for filename in control_filenames:
        preprocess_control(filename)
        grid += np.loadtxt(filename[0:-len(args.ext)] + '.csv')
    gray_hist = np.sum(grid, axis=1)
    thresh_idx = safe_otsu(gray_hist)
    background_threshold = gray_bins[thresh_idx]
    print(f'processing controls took {time.time() - t1} seconds')
print(f'background_threshold is {background_threshold}')

# Use control images to set stain_threshold
stain_threshold = args.stain_threshold
if args.controls_folder is not None:
    rmb_cumsum = np.sum(grid[0:thresh_idx,:], axis=0)
    rmb_cumsum = np.cumsum(rmb_cumsum)
    rmb_cumsum /= rmb_cumsum[-1]
    thresh_idx = np.nonzero(rmb_cumsum > 1.0 - args.error_tolerance)[0][0]
    stain_threshold = rmb_bins[thresh_idx]
print(f'stain_threshold is {stain_threshold}')

# Process the images using these thresholds
for filename in filenames:
    if os.path.exists(filename[0:-len(args.ext)]) and not args.reprocess:
        print(f'already processed {filename}')
        continue
    print(f'processing {filename}')

    # Print a header in the output file denoting that we are starting to process this image
    output_file = open(output_filename, 'a')  # append mode
    output_file.write(f'{filename}\n')
    output_file.close()

    if not os.path.exists(filename[0:-len(args.ext)]):
        os.mkdir(filename[0:-len(args.ext)])
    slide = openslide.OpenSlide(filename)
    dims = slide.dimensions
    level = slide.get_best_level_for_downsample(50)
    downsample_factor = slide.level_downsamples[level]
    thumbnail = np.copy(np.asarray(slide.read_region(location=(0,0), level=level, size=slide.level_dimensions[level])))
    # Partition the slide into grid cells
    nx = dims[0] // args.region_size
    ny = dims[1] // args.region_size
    # Choose a random permutation of the grid cells
    order = np.random.permutation(nx * ny)
    x_coords = np.arange(0, dims[0]-args.region_size, args.region_size)
    y_coords = np.arange(0, dims[1]-args.region_size, args.region_size)

    fig = plt.figure(figsize=(nx/2, ny/2), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(thumbnail, aspect='auto')

    histfig = plt.figure()
    histax = plt.subplot(111)
    histfig.add_axes(histax)

    t1 = time.time()
    count = 1
    total_bins = None
    stained_fracs = []
    for region in order:
        if np.isin(region, exclude_dict[filename.split(os.path.sep)[-1]]):
            print(f'excluding region {region}')
            continue
        region_x = region // ny
        region_y = region - region_x * ny
        img = np.asarray(slide.read_region(location=(x_coords[region_x], y_coords[region_y]), level=0, size=(args.region_size, args.region_size)))
        rmb, background_fraction = red_minus_blue(img, include_background=args.include_background, background_threshold=background_threshold)
        if background_fraction > args.background_fraction:
            continue
        # Draw this region on the thumbnail
        ax.text(x_coords[region_x]//downsample_factor, y_coords[region_y]//downsample_factor, str(region), fontsize=16)
        rect = patches.Rectangle((x_coords[region_x]//downsample_factor, y_coords[region_y]//downsample_factor), args.region_size//downsample_factor, args.region_size//downsample_factor, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        plt.imsave(os.path.join(filename[0:-len(args.ext)], f'region_{region}.png'), img)
        if args.show_processed_images:
            pic = np.copy(img).astype(float)
            gray = np.mean(pic, axis=2)
            background_idx = gray > background_threshold
            pic[:,:,0] = (pic[:,:,0] - pic[:,:,2]) / np.maximum(1, pic[:,:,2])
            idx = (pic[:,:,0] > stain_threshold)
            mincol = min(pic[:,:,0].flatten())
            maxcol = max(pic[:,:,0].flatten())
            pic[:,:,0] = (pic[:,:,0] - mincol) * 200 / (maxcol - mincol)
            pic[:,:,1] = pic[:,:,0]
            pic[:,:,2] = pic[:,:,0]
            pic[idx,0] = pic[idx,0] + 55 # Make stained regions pink
            # Make whitespace regions green
            if not args.include_background:
                pic[background_idx,1] = pic[background_idx,1] + 55
            plt.imsave(os.path.join(filename[0:-len(args.ext)], f'region_{region}_rmb_normalized.png'), pic.astype('uint8'))
        n, _, _ = histax.hist(rmb, density=True, bins=bins, histtype='step', label=f'region_{region}')
        if total_bins is None:
            total_bins = n
        else:
            total_bins = total_bins + n
        np.savetxt(os.path.join(filename[0:-len(args.ext)], f'region_{region}.txt'), n/sum(n))

        stained_frac = len(rmb[rmb > stain_threshold]) / len(rmb)
        stained_fracs.append([region, stained_frac])

        output_file = open(output_filename, 'a')  # append mode
        output_file.write(f'region {region}: {stained_frac}\n')
        output_file.close()

        count = count + 1
        if count > args.num_regions:
            break
    t2 = time.time()
    print(f'processing regions took {t2-t1} seconds')
    fig.savefig(os.path.join(filename[0:-len(args.ext)], f'thumbnail.png'))

    # Process the full slide
    if args.include_full:
        t1 = time.time()
        # Partition the slide into grid cells
        nx = 16
        ny = 16
        xlen = dims[0] // 16
        ylen = dims[1] // 16
        # Iterate through the regions
        order = np.arange(nx*ny)
        x_coords = np.arange(0, dims[0], xlen)
        y_coords = np.arange(0, dims[1], ylen)
        bin_totals = None
        stained_totals = 0
        totals = 0
        for region in order:
            region_x = region // ny
            region_y = region - region_x * ny
            img = np.asarray(slide.read_region(location=(x_coords[region_x], y_coords[region_y]), level=0, size=(xlen, ylen)))
            rmb, _ = red_minus_blue(img, include_background=args.include_background, background_threshold=background_threshold)
            if len(rmb) > 0:
                n, _, _ = plt.hist(rmb, density=True, bins=bins)
                if bin_totals is None:
                    bin_totals = n
                else:
                    bin_totals = bin_totals + n
                stained_totals += len(rmb[rmb > stain_threshold])
                totals += len(rmb)
        n, _, _ = histax.hist(x=bins[:-1], density=True, bins=bins, histtype='step', label=f'full image', weights=bin_totals)
        np.savetxt(os.path.join(filename[0:-len(args.ext)], f'full.txt'), n/sum(n))
        stained_fracs.append([0, stained_totals / totals])

        output_file = open(output_filename, 'a')  # append mode
        output_file.write(f'full slide: {stained_totals / totals}\n')
        output_file.close()

        t2 = time.time()
        print(f'processing the full image took {t2-t1} seconds')
    n, _, _ = histax.hist(x=bins[:-1], density=True, bins=bins, histtype='step', label=f'all regions', weights=total_bins)
    np.savetxt(os.path.join(filename[0:-len(args.ext)], f'all_regions.txt'), n/sum(n))
    if len(stained_fracs) > 0:
        np.savetxt(os.path.join(filename[0:-len(args.ext)], f'stained_fractions.txt'), np.asarray(stained_fracs), fmt=['%5u', '%10.5f'])
    histax.legend()
    histax.set_xlabel('normalized red - blue')
    histfig.savefig(os.path.join(filename[0:-len(args.ext)], 'histogram.png'))
