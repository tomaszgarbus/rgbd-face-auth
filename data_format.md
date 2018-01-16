# Data format

## RGB photos
RGB photos should be stored in .png files.

## Depth or IR photos
* bytes 0-3: magic const, either `"PHDE"` or `"PHIR"`
* bytes 4-7: picture width as `uint32_t`
* bytes 8-11: picture height as `uint32_t`
* bytes 12+: width * height `float`s (4 bytes each) containing pixel values, row by row
