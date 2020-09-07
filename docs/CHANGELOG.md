# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.3] - 2020-09-07
### Changed
- Updated and improved IdentifiedCollection, now backed by a Dictionary for easier lookup

## [0.5.2] - 2020-07-05
### Changed
- Removes the temporary fix used when identifying plate centres

## [0.5.1] - 2020-06-07
### Fixed
- RGB images are correctly handled during colour mode conversion

## [0.5.0] - 2020-06-05
### Added
- Utilize multiprocessing for grouping and filtering `Timepoint`s to Colonies
- `silent` command line argument to prevent output to console
- `animation` command line argument to output
- New class function, from_path, for building an ImageFileCollection
### Changed
- Improved plot labels/legends
- `plots` command line argument replaced with `no-plots` switch
- `multiprocessing` command line argument replaced with `single-process` switch
- Reworked command line arguments, using simplified switches for boolean arguments
- Reworked command line arguments to follow GNU conventions
- Refactored `main` module
- Improve access model and remove forward typing
### Fixed
- Ensure image files are converted to RGB

## [0.4.4] - 2020-06-03
### Fixed
- Plots are now saved to the correct directory again

## [0.4.3] - 2020-06-02
### Added
- `--version` command line argument to display installed package version number
- `--image_formats` command line argument to display supported image formats
- Package configuration and now consolidated and loaded from a settings module
### Changed
- Improved the output from the `--help` command line argument to make it a little clearer

## [0.4.2] - 2020-05-18
### Added
- Support for `bmp` image files
### Fixed
- Wrong type reference in colony.colonies_from_timepoints
### Changed
- `image_file._load_image` will now retry once if it fails
## [0.4.1] - 2020-05-11
### Added
- Dockerfile and utility scripts
- Docker image is now available: https://hub.docker.com/r/erikwhite/colonyscanalyser
## [0.4.0] - 2020-03-26
### Added
- `plate_labels` command line argument
- `base` module with base classes to provide core functionality throughout the package
- `geometry` module with base shape classes
- `plate` module with `Plate` and `PlateCollection` classes
- `image_file` module with `ImageFile` and `ImageFileCollection` classes
- `growth_curve` module to provide curve fitting and parameters
- `plot_plate_images_animation` outputs animated gifs for each plate in two sizes
- Output a summary of plate properties to `plates_summary.csv`
- Full type hinting for all modules
### Changed
- Extended compatibility to Python 3.8
- Cached data is now not used by default
- Individual plots for each plate are now output by default
- `use_saved` command line argument renamed to `use_cached_data`
- Compressed serialised data filename changed to `cached_data`
- `save_plots` command line argument renamed to `plots`
- `plate_size` now defaults to 90 mm, the visible size of a 100 mm plate in a plate holder
- `plate_edge_cut` is now a percentage of the plate diameter, instead of a fixed pixel value
- Refactored most of the functions from `main` as static methods in the `plate` or `image_file` modules
- Improved flexibility when detecting date and time stamp information in file names
- Replaced growth properties in `Colony`, now implemented as `growth_curve.GrowthCurveModel`
- Modified plots to use new `GrowthCurveModel` properties
- Improved colony segmentation in low contrast images and under poor growth conditions
- Grouping of colony Timepoints is now faster and more accurate
- Colonies are no longer identified based on circularity
### Fixed
- A rare error when opening images using skimage.io.imread
- Corrected default DPI settings and conversion factor
### Removed
- A number of Colony properties - now implemented by `growth_curve.GrowthCurveModel`

## [0.3.4] - 2020-01-18
### Added
- plate_edge_cut command line argument
- Plate and colony ID map to show how they have been identified
### Changed
- Add border exclusion and slightly relax colony circularity filtering in segment_image

## [0.3.3] - 2019-12-19
### Added
- Colony colour identification and grouping
- Webcolors package and rgb_to_name function to provide CSS colour groupings
### Fixed
- crop_image will now correctly handle images without an alpha channel

## [0.3.2] - 2019-11-11
### Added
- Documentation
- Published documentation to GitHub pages (https://erik-white.github.io/ColonyScanalyser/)
- Unit tests for the colony module

## [0.3.1] - 2019-11-04
### Fixed
- Adjust setup to correctly find packages in implicit namespaces

## [0.3.0] - 2019-11-04
### Added
- Added changelog
### Changed
- Update package to use src structure
- Update setup for readme compatability with PyPi

## [0.2.2] - 2019-11-02
### Added
- GitHub action for automatically linting and testing pushes
- GitHub action for building and releasing package to PyPi
### Fixed
- Linting errors highlighted by flake8

## [0.2.1] - 2019-10-31
### Added
- Graceful exit if no colonies are found
- Workaround function to ensure correct plates are found in images
### Changed
- Improve Timepoint grouping by using distance comparison instead of rounding
- Updated Scikit-image to v0.16
### Removed
- Depreciated Tk import
- Removed depreciated regionprop coordinates

## [0.2.0] - 2019-10-28
### Added
- Multiprocessing: greatly improves image processing speed
- Now shows a progress bar when processing images
- Snyk security checks for dependencies
### Changed
- Per image processing: now processes a single image at a time
- Improve colony filtering, removes virtually all merged colonies
- Updated readme with images and code examples
### Fixed
- Greatly reduced memory usage by using per-image processing
- Filter out system files when locating images to process
- Rare divide by zero error when processing colony object data

## [0.1.2] - 2019-10-13
Inital release
### Added
- Image processing, plotting and data aggregation
- Python package uploaded to PyPi