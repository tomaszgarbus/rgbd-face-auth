# Libkinect

The `libkinect.hpp` file contains a library which provides a simplified intefrace
for using both Kinect v1 and v2 with libfreenect and libfreenect2.

## Requirements

* CMake >= 3.8.2
* [libfreenect](https://github.com/OpenKinect/libfreenect)
* [libfreenect2](https://github.com/OpenKinect/libfreenect2)
* [OpenCV](https://docs.opencv.org/ref/master/d7/d9f/tutorial_linux_install.html)
* [wxWidgets](https://www.wxwidgets.org/)
* zlib (`sudo apt-get install zlib1g-dev`)

## Available programs

* `live_display` - live Kinect display, shows RGB/depth/IR feed, allows saving
  frames to hard drive.
* `file_display` - shows depth/IR files saved by `live_display`.
* `thumbnailer` - allows showing thumbnails of depth/IR files in graphical file
  managers.

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Thumbnailer installation

You need to build the thumbnailer first, check the "Building" section above.

```bash
cd thumbnailer
xdg-mime install depth-ir-mime.xml
sudo cp depth-ir.thumbnailer /usr/share/thumbnailers/
sudo cp ../build/thumbnailer /usr/bin/depth-ir-thumbnailer
```
