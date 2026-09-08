Initial releases of the package were still based upon use as a library for 
personal use. As such, the package was not well documented and the structure
underwent some significant changes. The following is a summary of the changes
that have been made to the package beginning with the first PyPi release. 

Beginning with Version 0.3.1 the field analysis package was introduced, which 
may lead to broader interest from potential users. As such, I began to improve
documentation and testing with that release. 

Backwards compatibility has not been uniformly maintained by releases due to 
the primary impact being on my individual use. Beginning with version 0.4.0, 
the package will adopt a consistent approach and will strive to maintain 
backwards compatibility for releases within the same minor version (0.X.X). 

# Version 0.2 - Jun 15, 2022
First public release of the package via PyPi
# Version 0.2.1 - Jul 17, 2022
- Add wrapper for `pvlib.clearsky_index` to handle pandas type
# Version 0.2.2 - July 27, 2022
- Change input to camfilter to handle references that don't coincide with the 
site itself. 
- ** Breaks backwards compatibility **
# Version 0.2.3
- Add methods for calculating delay between signals to `signalproc`
# Version 0.2.4 - Oct 19, 2023
- Add some additional options to `cmv` for qc and filtering of the data
# Version 0.2.5
- Add location analysis in `field` package
# Version 0.3.1 - Nov 8, 2023
- Expands `field` analysis capability
- Add comprehensive testing
- ** Breaks backwards compatibility **
# Version 0.3.2 - Nov 14, 2023
- Major speed improvements via vectorization of CMV 
- ** Breaks backwards compatibility due to function signature changes **
# Version 0.3.3 - Nov 17, 2023
- Major speed improvements to field analysis (vectorization of transfer function delays)
- ** Breaks backwards compatibility **
# Version 0.3.4 - Nov 21, 2023
- Bug fix for handling and excluding NaN within `field`
# Version 0.3.5 - Dec 18, 2023
- Add multi-column support to several functions in `stats`. Thanks to Scott Sheppard for the suggestion.
# Version 0.4.0 - Apr 10, 2024
- Added some functions to `field` to support automated remapping.
    - See `field_reassignment_demo` for details.
- Added a function to `cmv` for identifying an optimum subset of CMV pairs. 
    - See `automate_cmv_demo` for details.
- Added additional statistics to the `jamaly` method's output within `cmv.compute_cmv`
- Added additional sample plant data for demonstration purposes. 
    - We now include two sample plants and sample data from the HOPE-Melpitz campaign at 1s and 10s resolution.
- Added additional demos, including Jupyter notebooks in some cases.
    - `automate_cmv_demo` - Automated CMV pre-processing with HOPE-Melpitz data, includes a Jupyter notebook.
    - `field_reassignment_demo` - Automated remapping of the field data, includes a Jupyter notebook.
    - `field_demo_full_process` - Overview of the full field processing workflow 
    - `field_demo_full_process_multithread` - Same as previous but including multithreading support for speed
- Added tests of these new functions for more completeness.
# Version 0.4.1 - Aug 8, 2024
- Bug fix for `cmv` so that `jamaly` r_corr is always positive. 
- Demo for digitizing a plant equipment layout added by @williamhobbs
- Improve comments in `cmv_demo`
- Add sphinx documentation for entire library
# Version 0.4.2 - Aug 8 - 2024
- Rename the library to solarspatialtools
# Version 0.4.3 & 0.4.4 - Aug 8, 2024
- Fixes to GitHub deployment issues, no functional changes
# Version 0.4.5 - Sept 13, 2024
- Improvements to documentation
- Simplification of github actions
- Added matplotlib dependency
- Move demos back to the parent directory using nblink
# Version 0.4.6 - Sept 13, 2024
- Fix license name in cfg
# Version 0.5.0 - Sept 15, 2024
- Updating dependencies  and test targets
- Fix deprecation warnings raised by numpy
- JOSS publication
# Version 0.5.1 - Nov 7, 2024
- Create `synthirrad` package
- Implement the `synthirrad.cloudfield` module
- Tests and demos for the new package
# Version 0.5.2 - June 22, 2026
- Implement the `synthirrad.copula` method based on Widen and Munkhammar (2019)
- Tests, demos and docs for the new method
- Bugfix - `spatial.latlon2lcs` was not handling arrays
- Introduce `spatial.lla2flat` and `spatial.spacetime_distances` to support copula implementation
- # Version 0.5.2a - June 22, 2026
- Fix failed documentation build.
- # Version 0.5.3 - Aug 4, 2026
- Adjust dt calculations in `stats` package to use median.
- # Version 0.5.4 - Aug 13, 2026
- Bugfix for CMV calculation in `cmv` package to handle NaN correlation values.
- # Version 0.5.5 - Aug 14, 2026
- Add tf-based delay option to CMV.
- Improve repr of Windspeeddata class. 
- # Version 0.5.6 - Sept 8, 2026
- Add degree calculation to `cmv` outdata for graph coverage of CMV pairs.