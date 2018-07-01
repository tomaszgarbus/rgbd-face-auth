# Common
This directory contains helpers shared across all `face_auth` code.

## `constants.py`
You can customize the constants locally, for your needs. They are described in comments, but
two constants you should pay attention to before running the code are `SHOW_PLOTS` and `DB_LOCATION`.

## `db_helper.py`
Introduces two classes `Database` and `DBHelper`.
* `DBHelper` locates and returns a list of found databases, as `Database` objects.
* `Database` is responsible for loading single database.

## `display_photo.py`
A simple script displaying a valid `.ir` or `.depth` file. Execute it with `python3.6 display_photo.py path_to_file`

## `tools.py`
Simple helpers mainly for image manipulation.