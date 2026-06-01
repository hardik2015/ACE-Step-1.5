#!/usr/bin/env python3
"""One-time helper: mint a Google Drive OAuth token for the Kaggle daily pipeline.

The Kaggle kernel uploads finished songs to a PERSONAL "My Drive" folder. Service
accounts cannot do that (0-byte My-Drive quota -> storageQuotaExceeded), so we
authenticate as YOU via an OAuth "authorized user" refresh token instead.

Run this ONCE on your own machine (it opens a browser for consent), then paste
the printed JSON into the Kaggle Secret `GDRIVE_OAUTH_TOKEN`.

Prerequisites
-------------
1. Google Cloud Console -> create / pick a project.
2. Enable the "Google Drive API" for that project.
3. APIs & Services -> Credentials -> Create credentials -> OAuth client ID ->
   Application type = "Desktop app". Download the JSON (client_secret_*.json).
4. OAuth consent screen: add your own Google account under "Test users" (a
   Testing-status app issues refresh tokens fine for your own account).

Usage
-----
    pip install google-auth-oauthlib google-auth
    python scripts/gdrive_oauth_setup.py path/to/client_secret.json

Then:
    - copy the printed JSON  -> Kaggle kernel -> Add-ons -> Secrets ->
      GDRIVE_OAUTH_TOKEN
    - put your target folder id (the part after /folders/ in the Drive URL) ->
      Kaggle Secret GDRIVE_FOLDER_ID
"""
import argparse
import sys

# Full drive scope so the token may upload into a pre-existing folder by id that
# this app did not itself create (drive.file would not allow that).
SCOPES = ["https://www.googleapis.com/auth/drive"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("client_secret",
                    help="Path to the OAuth 'Desktop app' client_secret JSON")
    ap.add_argument("--port", type=int, default=0,
                    help="Local redirect port (default: auto)")
    args = ap.parse_args()

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Install deps first:\n    pip install google-auth-oauthlib google-auth",
              file=sys.stderr)
        return 2

    flow = InstalledAppFlow.from_client_secrets_file(args.client_secret, SCOPES)
    # Opens a browser, asks for consent, captures the code on a local redirect.
    creds = flow.run_local_server(port=args.port, prompt="consent")

    if not creds.refresh_token:
        print("\nERROR: Google did not return a refresh_token. Re-run; if it keeps "
              "happening, revoke the app at https://myaccount.google.com/permissions "
              "then try again (consent must be re-granted to get a refresh token).",
              file=sys.stderr)
        return 1

    token_json = creds.to_json()  # authorized-user JSON (client_id/secret/refresh_token)
    print("\n" + "=" * 70)
    print("SUCCESS. The kernel accepts EITHER of these on Kaggle (Add-ons -> Secrets):")
    print("=" * 70)
    print("\nOption 1 -- one bundled secret  GDRIVE_OAUTH_TOKEN  =")
    print(token_json)
    print("\nOption 2 -- three separate secrets:")
    print(f"  GDRIVE_REFRESH_TOKEN  = {creds.refresh_token}")
    print(f"  GDRIVE_CLIENT_ID      = {creds.client_id}")
    print(f"  GDRIVE_CLIENT_SECRET  = {creds.client_secret}")
    print("=" * 70)
    print("Keep these secret. They grant write access to your Google Drive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
