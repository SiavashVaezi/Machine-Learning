from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# راه‌اندازی WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
driver.set_script_timeout(60)

driver.get('https://bama.ir/car')

wait = WebDriverWait(driver, 10)

# اسکرول تا انتهای صفحه
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# منتظر بارگذاری کارت‌ها
try:
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a.bama-ad.listing')))
except:
    print("⛔ کارت‌ها بارگذاری نشدند")

cars = []
car_cards = driver.find_elements(By.CSS_SELECTOR, 'a.bama-ad.listing')

for card in car_cards:
    try:
        # عنوان خودرو
        try:
            title = card.find_element(By.CSS_SELECTOR, 'p.bama-ad__title').text.strip()
        except:
            title = "Unknown"

        # سال تولید (اولین span داخل جزئیات)
        try:
            year = card.find_element(By.CSS_SELECTOR, 'div.bama-ad__detail-row span').text.strip()
        except:
            year = "Unknown"

        # کارکرد (span با کلاس خاص)
        try:
            kilometer = card.find_element(By.CSS_SELECTOR, 'span.dir-ltr').text.strip()
        except:
            kilometer = "Unknown"

        #قیمت

        try:
            price = card.find_element(By.CSS_SELECTOR, 'div.bama-ad__price-holder').text.strip()
        except:
            price = "Unknown"

    except Exception as e:
        print("⚠️ خطا در استخراج اطلاعات:", e)
        title = "Unknown"
        year = "Unknown"
        kilometer = "Unknown"
        price="Unknown"

    cars.append({
        'Title': title,
        'Production Year': year,
        'Kilometer': kilometer,
        'Price': price
    })

driver.quit()

# ذخیره در فایل Excel
df = pd.DataFrame(cars)
df.to_excel('cars.xlsx', index=False)

print("✅ استخراج اطلاعات با موفقیت تمام شد. فایل cars.xlsx ساخته شد.")



