import requests
import tqdm
import os
import zipfile
import errno


def ensure_directory_exists(path: str):
    try:
        # `exist_ok` option is only available in Python 3.2+
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def unzip_file(src_path, dest_path, filename):
    """unzips file located at src_path into destination_path"""
    print("unzipping file...")

    # construct full path (including file name) for unzipping
    unzip_path = os.path.join(dest_path, filename)
    ensure_directory_exists(unzip_path)

    # extract data
    with zipfile.ZipFile(src_path, "r") as z:
        z.extractall(unzip_path)

    return True

def download_file(url: str, dest_path: str, show_progress_bars: bool = True):
    file_size = 0
    req = requests.get(url, stream=True)
    req.raise_for_status()

    # Total size in bytes.
    total_size = int(req.headers.get('content-length', 0))

    if os.path.exists(dest_path):
        print("target file already exists")
        file_size = os.stat(dest_path).st_size  # File size in bytes
        if file_size < total_size:
            # Download incomplete
            print("resuming download")
            resume_header = {'Range': 'bytes=%d-' % file_size}
            req = requests.get(url, headers=resume_header, stream=True,
                               verify=False, allow_redirects=True)
        elif file_size == total_size:
            # Download complete
            print("download complete")
            return
        else:
            # Error, delete file and restart download
            print("ERROR: deleting file and restarting")
            os.remove(dest_path)
            file_size = 0
    else:
        # File does not exist, starting download
        print("starting download")

    # write dataset to file and show progress bar
    pbar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True,
                     desc=dest_path, disable=not show_progress_bars)
    # Update progress bar to reflect how much of the file is already downloaded
    pbar.update(file_size)
    with open(dest_path, "ab") as dest_file:
        for chunk in req.iter_content(1024):
            dest_file.write(chunk)
            pbar.update(1024)
