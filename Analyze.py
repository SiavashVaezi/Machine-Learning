import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = r"C:\Windows\Fonts\Tahoma.ttf"
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
# 1. خواندن فایل اکسل
df = pd.read_excel("cars.xlsx")

# 2. پاک‌سازی عنوان و استخراج برند و مدل
df['Title'] = df['Title'].astype(str).str.strip()
df['Brand'] = df['Title'].str.split('،| ').str[0]
df['Model'] = df['Title'].str.split('،| ').str[1:].str.join(' ')

# 3. حذف آگهی‌های پیش‌فروش یا حواله
df['IsPreSale'] = df['Title'].str.contains('پیش.?فروش|حواله', case=False, na=False)
df = df[~df['IsPreSale']].copy()

# 4. تبدیل قیمت به عدد
def parse_price(p):
    if isinstance(p, str):
        digits = p.replace(',', '').strip()
        if digits.isdigit():
            return int(digits)
    return None

df['Price'] = df['Price'].apply(parse_price)

# 5. تبدیل کیلومتر به عدد
def parse_kilometer(km):
    if isinstance(km, str):
        if 'صفر' in km:
            return 0
        km = km.replace(',', '').replace('km', '').replace('کیلومتر', '').strip()
        if km.isdigit():
            return int(km)
    return None

df['Kilometer'] = df['Kilometer'].apply(parse_kilometer)

# 6. تبدیل سال تولید (میلادی به شمسی)
def convert_year(year):
    try:
        year = int(year)
    except:
        return None
    if year > 1500:  # سال میلادی
        return year - 621
    elif 1300 <= year <= 1500:  # سال شمسی
        return year
    else:
        return None

df['Production Year'] = df['Production Year'].apply(convert_year)

# 7. حذف ردیف‌های ناقص
df = df.dropna(subset=['Price', 'Kilometer', 'Production Year'])

# 8. محاسبه سن خودرو
df['CarAge'] = 1403 - df['Production Year']

# 9. ساخت ویژگی‌ها و هدف
features = df[['Brand', 'Model', 'Kilometer', 'CarAge']].copy()
target = df['Price']

# 10. رمزگذاری ویژگی‌های متنی
le_brand = LabelEncoder()
le_model = LabelEncoder()
features['Brand'] = le_brand.fit_transform(features['Brand'].astype(str))
features['Model'] = le_model.fit_transform(features['Model'].astype(str))

# 11. تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 12. آموزش مدل
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 13. ارزیابی مدل
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🎯 میانگین خطای مطلق (MAE):", round(mae))
print("📊 ضریب تعیین (R²):", round(r2, 3))

# 14. اهمیت ویژگی‌ها
importances = model.feature_importances_
feature_names = features.columns

print("\n🏆 اهمیت ویژگی‌ها:")
for name, score in zip(feature_names, importances):
    print(f"{name}: {round(score * 100, 1)}٪")

# --- نمودار 1: اهمیت ویژگی‌ها ---
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances)
plt.title("Importance of Features in Price Prediction")
plt.xlabel("Impotance")
plt.ylabel("Feature")
plt.grid(True)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

# --- نمودار 2: پراکندگی قیمت واقعی و پیش‌بینی ---
plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual and predicted price dispersion")
plt.grid(True)
plt.tight_layout()
plt.savefig("Actual and predicted price dispersion.png", dpi=300)
plt.show()

# --- نمودار 3: توزیع قیمت‌ها ---
plt.figure(figsize=(8,5))
plt.hist(df['Price'], bins=50, color='purple', alpha=0.7)
plt.xlabel("Price")
plt.ylabel("No. of Cars")
plt.title("Distribution of car prices")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("Distribution of car prices.png", dpi=300)
plt.show()

# --- نمودار 4: رابطه سن خودرو و قیمت ---
plt.figure(figsize=(8,5))
plt.scatter(df['CarAge'], df['Price'], alpha=0.3, color='orange')
plt.xlabel("Car age (years)")
plt.ylabel("Pric")
plt.title("Relationship between car age and price")
plt.grid(True)
plt.tight_layout()
plt.savefig("Relationship between car age and price.png", dpi=300)
plt.show()

# --- نمودار 5: میانگین قیمت بر اساس برند ---
plt.figure(figsize=(10,6))
avg_price_brand = df.groupby('Brand')['Price'].mean().sort_values(ascending=False).head(15)
avg_price_brand.plot(kind='bar', color='green', alpha=0.7)
plt.ylabel("Average of Price")
plt.title("Average car prices by brand")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("Average car prices by brand.png", dpi=300)
plt.show()
