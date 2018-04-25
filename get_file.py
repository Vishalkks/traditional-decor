from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

file_id = '1WcQcgmLMktI6_uW0o6sZgCGDGnh67ih-'

import io
from io import BytesIO   
from googleapiclient.http import MediaIoBaseDownload

from googleapiclient.discovery import build
drive_service = build('drive', 'v3')

request = drive_service.files().get_media(fileId=file_id)
downloaded = io.BytesIO()
downloader = MediaIoBaseDownload(downloaded, request)
done = False
while done is False:
  status, done = downloader.next_chunk()
  if status:
      print("Download %%%d%%." % int(status.progress() * 100))
  print("Download Complete!")

downloaded.seek(0)

with open('/tmp/model.h5', 'wb') as f:
  f.write(downloaded.read())
  f.close()
