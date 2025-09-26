import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

data = {
    'Jumlah_Iklan': [10,15,12,20,25,18,22,8,28,16],
    'Durasi_Iklan': [15,20,25,30,35,25,30,20,40,25],
    'Jumlah_Pembeli': [150,160,155,180,200,175,190,140,210,170]
}
df = pd.DataFrame(data)

X = df[['Jumlah_Iklan', 'Durasi_Iklan']]
y = df['Jumlah_Pembeli']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Intercept:", model.intercept_)
print("Koefisien:", list(zip(X.columns, model.coef_)))
print("MSE:", mean_squared_error(y_test, y_pred))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Jumlah_Iklan'], df['Durasi_Iklan'], y, c='blue', s=50, label='Data Asli')

# Membuat grid untuk bidang prediksi
x_surf, y_surf = np.meshgrid(
    np.linspace(df['Jumlah_Iklan'].min(), df['Jumlah_Iklan'].max(), 20),
    np.linspace(df['Durasi_Iklan'].min(), df['Durasi_Iklan'].max(), 20)
)
z_surf = model.intercept_ + model.coef_[0]*x_surf + model.coef_[1]*y_surf
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.4)

ax.set_xlabel('Jumlah Iklan')
ax.set_ylabel('Durasi Iklan (detik)')
ax.set_zlabel('Jumlah Pembeli')
ax.set_title('Regresi Linier Berganda - Bidang Prediksi')
plt.tight_layout()
plt.show()

plt.scatter(df['Jumlah_Iklan'], y, color='blue')
# prediksi jika Durasi_Iklan rata-rata
durasi_mean = df['Durasi_Iklan'].mean()
plt.plot(
    df['Jumlah_Iklan'],
    model.intercept_ + model.coef_[0]*df['Jumlah_Iklan'] + model.coef_[1]*durasi_mean,
    color='red'
)
plt.title('Jumlah Pembeli vs Jumlah Iklan (Durasi tetap rata-rata)')
plt.xlabel('Jumlah Iklan')
plt.ylabel('Jumlah Pembeli')
plt.show()

plt.scatter(df['Durasi_Iklan'], y, color='green')
iklan_mean = df['Jumlah_Iklan'].mean()
plt.plot(
    df['Durasi_Iklan'],
    model.intercept_ + model.coef_[0]*iklan_mean + model.coef_[1]*df['Durasi_Iklan'],
    color='red'
)
plt.title('Jumlah Pembeli vs Durasi Iklan (Jumlah Iklan tetap rata-rata)')
plt.xlabel('Durasi Iklan (detik)')
plt.ylabel('Jumlah Pembeli')
plt.show()

contoh = pd.DataFrame({'Jumlah_Iklan':[20], 'Durasi_Iklan':[30]})
print("Prediksi jumlah pembeli (20 iklan, 30 detik):", model.predict(contoh))
