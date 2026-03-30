import csv
import requests
import os
from urllib.parse import urlparse, unquote

def sanitize_filename(name):
    """Remove invalid characters from filename"""
    invalid = '<>:"/\\|?*'
    for char in invalid:
        name = name.replace(char, '_')
    return name.strip()

def get_clean_filename(url, order):
    """Generate a nice filename from the URL (add .pdf if missing)"""
    parsed = urlparse(url)
    # Get the last part of the path and decode any % encodings
    basename = os.path.basename(unquote(parsed.path))
    
    # Fallback if URL ends with something weird (some old Tesla links)
    if not basename or basename in ['image', 'upload', 'IR', '']:
        basename = f"TSLA_Update_{order}"
    
    # Force .pdf extension for links that don't have one (common on older files)
    if not basename.lower().endswith('.pdf'):
        basename += '.pdf'
    
    return sanitize_filename(basename)

def download_tesla_pdfs(csv_file='/Users/alanwong/development/machine-learning-for-trading/OtherProjects/tesla.csv'):
    """Main function - downloads only the "Download" PDF links"""
    folder = 'tesla_shareholder_updates'
    os.makedirs(folder, exist_ok=True)
    
    # Good User-Agent so Tesla's servers don't block us
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
    }
    
    count = 0
    failed = 0
    
    print("📥 Starting download of Tesla Shareholder Update PDFs...\n")
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ONLY download rows where the column says "Download" (skip all 10-K/10-Q .htm files)
            if row.get('pdf', '').strip() != 'Download':
                continue
                
            url = row['pdf-href'].strip()
            order = row['web-scraper-order'].strip()
            
            if not url or not url.startswith('http'):
                continue
            
            filename = get_clean_filename(url, order)
            filepath = os.path.join(folder, filename)
            
            # Skip if we already downloaded it
            if os.path.exists(filepath):
                print(f"✅ Already exists: {filename}")
                count += 1
                continue
            
            try:
                print(f"⬇️  Downloading: {filename}")
                response = requests.get(url, headers=headers, stream=True, timeout=45)
                
                # If we get a 403 (like from Akamai Bot Manager), we can try curl_cffi
                if response.status_code == 403:
                    try:
                        from curl_cffi import requests as cffi_requests
                        print(f"⚠️  Got 403 Forbidden. Attempting bypass using curl_cffi for {filename}...")
                        response = cffi_requests.get(url, impersonate="chrome", stream=True, timeout=45)
                    except ImportError:
                        pass # Fall through to raise_for_status which will throw the 403 Exception
                        
                response.raise_for_status()
                
                with open(filepath, 'wb') as f_out:
                    for chunk in response.iter_content(chunk_size=8192 * 4):
                        if chunk:
                            f_out.write(chunk)
                
                print(f"✅ Saved: {filename}")
                count += 1
                
            except Exception as e:
                print(f"❌ Failed {filename}: {e}")
                if "403" in str(e):
                    print(f"   💡 TIP: This URL is protected by Akamai Bot Manager.")
                    print(f"   To automate this download, install curl_cffi: pip install curl_cffi")
                    print(f"   Otherwise, manually download it from: {url}")
                failed += 1
    
    print("\n" + "="*50)
    print("🎉 DOWNLOAD COMPLETE!")
    print(f"✅ Successfully downloaded: {count} PDF files")
    if failed > 0:
        print(f"❌ Failed downloads: {failed}")
    print(f"📁 Files saved in folder: ./{folder}/")
    print("="*50)

if __name__ == "__main__":
    # How to use:
    # 1. Copy ALL the CSV content from the message (including the header line) 
    #    and save it as a file named tesla_links.csv in the same folder as this script.
    # 2. Run: python download_tesla_pdfs.py
    download_tesla_pdfs()