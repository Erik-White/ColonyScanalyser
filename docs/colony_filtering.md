# Filtering criteria
<img align="right" src="../images/plate_timelapse_tiny.gif">
Images are first segmented to find objects in the image that appear to be colonies. Objects are assessed on three initial criteria:

- Size: objects smaller than 1.5 pixels are ignored. This reduces that amount of noise at the limit of detection
- Plate edges: colonies are removed if they grow over the edge of the plate measurement area
- Static objects: objects or image arefacts that are present in the first (i.e. empty) plate are removed. Colonies that intersect with these objects are also removed

After segmenting, the resulting objects are filtered further by three criteria:

- Colonies that do not have enough data points
- Colonies with large gaps in the data points
- Objects that do not show growth
- Colonies with large initial areas (merged colonies)

This filtering has the effect of removing colonies that merge together. This is beneficial as merged colonies do not exhibit the same growth characteristics as unimpinged colonies. It is also practically impossible to divide a merged area between two individual colonies.