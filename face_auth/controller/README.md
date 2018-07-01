# Controller
## `normalization.py`
Controller-layer code for face normalization. It exposes three functions:
* `preprocessing` - trims the face, normalizes depth by mean and std. dev., locates
   interesting points on the face (such as eyebrows or chin) and stores them in
   `Face` object.
* `normalized` - does the preprocessing + optionally (if `rotate = True`) rotates
  the face to frontal position
* `hog_and_entropy` - computes HOGs and entropy maps