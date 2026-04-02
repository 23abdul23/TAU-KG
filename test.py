from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# CONFIG
PMC_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12782617/"
OUTPUT_FILE = "pmc_page.html"

# Setup options
options = Options()
options.add_argument("--headless=new")

# Auto-download + manage driver
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# Load page
driver.get(PMC_URL)

# Wait
driver.implicitly_wait(5)

# Get HTML
html = driver.page_source

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

print("Saved HTML successfully")

driver.quit()