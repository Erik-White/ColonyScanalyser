# Filtering criteria
<img align="right" src="images/plate_timelapse_tiny.gif">
Images are first segmented to find objects in the image that appear to be colonies. Objects are assessed on three initial criteria:

- Size: objects smaller than 5 pixels are ignored. This is to reduce the level of background noise that would otherwise be picked up
- Eccentricity: objects with high eccentricity are ignored as colonies are always circular
- Circularity: along with eccentricity, this gives a good indication of the 'roundness' of an object

This filtering has the effect of removing colonies that merge together. This is beneficial as merged colonies do not exhibit the same growth characteristics as unimpinged colonies. It is also practically impossible to divide a merged area between two individual colonies.

After segmenting, the resulting objects are filtered further by three criteria:

- Colonies that do not have enough data points
- Objects that do not show growth
- Colonies with large initial areas (usually merged colonies)