import requests

def download_pdfs(links_file, download_folder):
    with open(links_file, 'r') as file:
        links = file.readlines()
    
    for link in links:
        link = link.strip()
        if link:
            try:
                response = requests.get(link)
                response.raise_for_status()
                pdf_name = link.split('/')[-1]
                pdf_path = f"{download_folder}/{pdf_name}"
                with open(pdf_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                print(f"Downloaded: {pdf_name}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {link}: {e}")

if __name__ == "__main__":
    links_file = ''  
    download_folder = 'orders' 
    download_pdfs(links_file, download_folder)