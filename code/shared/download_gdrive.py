import os
import re
import sys
import requests
from bs4 import BeautifulSoup


def get_original_filename(session, file_id):
    """Fetch the original filename from Google Drive using multiple methods."""

    # Method 1: Try the file view page title
    url = f"https://drive.google.com/file/d/{file_id}/view"
    response = session.get(url)
    match = re.search(r'<title>(.+?) - Google Drive</title>', response.text)
    if match:
        name = match.group(1).strip()
        if name and name != "Google Drive":
            return name

    # Method 2: Try parsing filename from meta tags or og:title
    match = re.search(r'<meta\s+property="og:title"\s+content="([^"]+)"', response.text)
    if match:
        name = match.group(1).strip()
        if name:
            return name

    # Method 3: Try a HEAD request on the download URL and check Content-Disposition
    dl_url = f"https://docs.google.com/uc?export=download&id={file_id}"
    response = session.get(dl_url, stream=True)
    cd = response.headers.get("Content-Disposition", "")
    # Match filename*=UTF-8''name or filename="name"
    match = re.search(r"filename\*=UTF-8''(.+?)(?:;|$)", cd)
    if match:
        from urllib.parse import unquote
        return unquote(match.group(1).strip())
    match = re.search(r'filename="(.+?)"', cd)
    if match:
        return match.group(1).strip()

    return None


def download_from_gdrive(file_id, output_dir=".", custom_name=None):
    session = requests.Session()

    # Use custom name or resolve original filename
    if custom_name:
        filename = custom_name
        print(f"Using custom filename: {filename}")
    else:
        filename = get_original_filename(session, file_id)
        if filename:
            print(f"Detected original filename: {filename}")
        else:
            filename = "downloaded_file"
            print("Could not detect original filename, using default.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Step 1: Initial request
    base_url = "https://docs.google.com/uc"
    params = {"export": "download", "id": file_id}
    response = session.get(base_url, params=params, stream=True)

    # Step 2: If Content-Disposition is present, the file is directly available (small files)
    if "content-disposition" not in response.headers:
        print("Large file detected, handling confirmation page...")

        # Try to get the token from cookies
        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

        if not token:
            # Parse the HTML confirmation page
            soup = BeautifulSoup(response.text, "html.parser")

            # Look for the download form (newer Google Drive flow)
            download_form = soup.find("form", {"id": "download-form"})
            if download_form and download_form.get("action"):
                download_url = download_form["action"]
                # Collect all hidden input fields (confirm, uuid, etc.)
                hidden_inputs = download_form.find_all("input", {"type": "hidden"})
                form_params = {}
                for inp in hidden_inputs:
                    if inp.get("name") and inp.get("value") is not None:
                        form_params[inp["name"]] = inp["value"]

                response = session.get(download_url, params=form_params, stream=True)
            else:
                # Fallback: search for confirm token in HTML
                match = re.search(r'confirm=([0-9A-Za-z_-]+)', response.text)
                if match:
                    token = match.group(1)
                    params["confirm"] = token
                    response = session.get(base_url, params=params, stream=True)
                else:
                    print("WARNING: Could not find confirmation token. Download may fail.")
                    params["confirm"] = "t"
                    response = session.get(base_url, params=params, stream=True)
        else:
            # Use cookie token
            params["confirm"] = token
            response = session.get(base_url, params=params, stream=True)

    # If we still have the default name, try extracting from the final response headers
    if filename == "downloaded_file":
        cd = response.headers.get("Content-Disposition", "")
        match = re.search(r"filename\*=UTF-8''(.+?)(?:;|$)", cd)
        if match:
            from urllib.parse import unquote
            filename = unquote(match.group(1).strip())
            print(f"Detected filename from download headers: {filename}")
            output_path = os.path.join(output_dir, filename)
        else:
            match = re.search(r'filename="(.+?)"', cd)
            if match:
                filename = match.group(1).strip()
                print(f"Detected filename from download headers: {filename}")
                output_path = os.path.join(output_dir, filename)

    # Download with progress
    total_size = int(response.headers.get("Content-Length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100
                    print(f"\rProgress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB", end="", flush=True)

    print()
    final_size = os.path.getsize(output_path)
    print(f"Downloaded to: {output_path} ({final_size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download a file from Google Drive.")
    parser.add_argument("file_id", help="Google Drive file ID")
    parser.add_argument("-o", "--output-dir", default=".", help="Directory to save the file to (default: current directory)")
    parser.add_argument("-n", "--name", default=None, help="Custom filename (default: original name from Google Drive)")

    args = parser.parse_args()
    download_from_gdrive(args.file_id, output_dir=args.output_dir, custom_name=args.name)