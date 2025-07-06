import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sayfa genişliğini manuel olarak ayarlama
st.set_page_config(
    page_title="Veri Bilimi Uygulaması",
    page_icon="📊",
    layout="wide",  # "centered" yerine "wide" kullan
    initial_sidebar_state="expanded"
)

# CSS ile sayfa genişliğini %95 olarak ayarlama ve daha etkili stil
st.markdown("""
<style>
    .main .block-container {
        max-width: 75% !important;
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        margin: 0 auto;
    }
    
    .stApp {
        max-width: 75% !important;
        margin: 0 auto;
    }
    
    .main {
        max-width: 75% !important;
        margin: 0 auto;
    }
    
    /* Ek stil ayarları */
    .stSlider {
        width: 100% !important;
    }
    
    .stSelectbox {
        width: 100% !important;
    }
    
    /* Container genişliği */
    .stContainer {
        max-width: 75% !important;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Ana sayfa ortasında parametre seçimi için container
with st.container():
    st.markdown("---")
    st.subheader('Veri Oluşturma Parametreleri')
    
    # İki sütunlu layout oluştur
    col1, col2 = st.columns(2)

    # Sol sütun
    with col1:
        # n_samples parametresi
        n_samples = st.slider(
            'Örnek Sayısı (n_samples):',
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help='Veri setindeki toplam örnek sayısını belirler'
        )
        
        # centers parametresi
        center_options = [[[-2, -2], [2, 2]], [[-3, -3], [3, 3]], [[-1, -1], [1, 1]]]
        center_index = st.slider(
            'Merkez Konumları (centers):',
            min_value=0,
            max_value=len(center_options)-1,
            value=0,
            help='Küme merkezlerinin konumları'
        )
        centers = center_options[center_index]

    # Sağ sütun
    with col2:
        # cluster_std parametresi
        cluster_std = st.slider(
            'Küme Standart Sapması (cluster_std):',
            min_value=0.1,
            max_value=10.0,
            value=3.0,
            step=0.1,
            help='Kümelerin dağılım genişliği. Düşük değerler daha sıkı kümeler oluşturur'
        )
# Train-test split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
import matplotlib.pyplot as plt
# Yapay veri oluşturma
X, y = make_blobs(
    n_samples=n_samples,
    centers=centers,
    n_features=2,
    cluster_std=cluster_std,
    random_state=42
)

# DataFrame oluşturma
df = pd.DataFrame(X, columns=['X1', 'X2'])
df['target'] = y

# Veriyi train ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree Classifier oluşturma ve eğitme
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Tahmin yapma
y_pred = dt_classifier.predict(X_test)

# Decision Boundary için mesh grid oluşturma
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Mesh grid üzerinde tahmin yapma
Z = dt_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# İki subplot oluşturma
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Özel renk haritası oluşturma (1=kırmızı, 0=mavi)
colors = ['blue', 'red']
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(colors)

# 1. Plot: Train verisi ile decision boundary
ax1.contourf(xx, yy, Z, alpha=0.4, cmap=custom_cmap)
scatter1 = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                      cmap=custom_cmap, s=50, edgecolors='black')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_title('Train Verisi ve Decision Boundary')
ax1.legend(handles=scatter1.legend_elements()[0], 
          labels=['negatif 0 (Mavi)', 'pozitif 1 (Kırmızı)'], title='Sınıflar')

# 2. Plot: Test verisi gerçek değerlerine göre renklendirilmiş
ax2.contourf(xx, yy, Z, alpha=0.4, cmap=custom_cmap)
scatter2 = ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                      cmap=custom_cmap, s=50, edgecolors='black')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_title('Test Verisi (Gerçek Değerler) ve Decision Boundary')
ax2.legend(handles=scatter2.legend_elements()[0], 
          labels=['negatif 0 (Mavi)', 'pozitif 1 (Kırmızı)'], title='Sınıflar')

# Test verisi üzerine TP, FP, FN, TN etiketlerini ekleme
for i, (x, y_coord, true_label, pred_label) in enumerate(zip(X_test[:, 0], X_test[:, 1], y_test, y_pred)):
    if true_label == 1 and pred_label == 1:
        label = 'TP'
    elif true_label == 0 and pred_label == 0:
        label = 'TN'
    elif true_label == 0 and pred_label == 1:
        label = 'FP'
    else:  # true_label == 1 and pred_label == 0
        label = 'FN'
    
    ax2.annotate(label, (x, y_coord), xytext=(0, 10), textcoords='offset points',
                fontsize=8, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
st.pyplot(plt)
# Confusion Matrix parametreleri (şablona uygun)
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
# cm[0,0]: Target 0 için TN, cm[0,1]: Target 0 için FP, cm[1,0]: Target 1 için FN, cm[1,1]: Target 1 için TP
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

# Confusion Matrix görselleştirme (şablona uygun)
plt.figure(figsize=(3, 2.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pozitif (1)', 'Negatif (0)'], 
            yticklabels=['Pozitif (1)', 'Negatif (0)'])
plt.title('Predicted Class')
plt.ylabel('Actual Class')

st.pyplot(plt, use_container_width=False)

st.image('classification_report.png', 
         caption='Sınıflandırma Raporu', 
         use_container_width =True)

# 1. Tanım
st.write("#### 1. Tanım:")
st.markdown("""
- **Positive/Negative**: Modelin tahmin ettiği sınıf (pozitif veya negatif)
- **True/False**: Tahminin doğru mu yanlış mı olduğu
- **TP (True Positive)**: Pozitif tahmin + Doğru tahmin
- **TN (True Negative)**: Negatif tahmin + Doğru tahmin
- **FP (False Positive)**: Pozitif tahmin + Yanlış tahmin
- **FN (False Negative)**: Negatif tahmin + Yanlış tahmin
""")

# 2. Tanım
st.write("#### 2. Tanım:")
st.markdown("""
- **TP (True Positive - Gerçek Pozitif)**: Modelin pozitif(1) olarak tahmin ettiği ve gerçekten de pozitif(1) olan örnekler
- **TN (True Negative - Gerçek Negatif)**: Modelin negatif(0) olarak tahmin ettiği ve gerçekten de negatif(0) olan örnekler  
- **FP (False Positive - Yanlış Pozitif)**: Modelin pozitif(1) olarak tahmin ettiği ama gerçekte negatif(0) olan örnekler (Tip I hata)
- **FN (False Negative - Yanlış Negatif)**: Modelin negatif(0) olarak tahmin ettiği ama gerçekte pozitif(1) olan örnekler (Tip II hata)
""")



# Metriklerin hesaplanması
accuracy = (TP + TN) / (TP + TN + FP + FN)
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
npv = TN / (TN + FN) if (TN + FN) > 0 else 0

st.write("### Metrik Formülleri ve Sonuçları:")
st.latex(fr"\Large\text{{TP (True Positive) = }} {TP}")
st.latex(fr"\Large\text{{TN (True Negative) = }} {TN}")
st.latex(fr"\Large\text{{FP (False Positive) = }} {FP}")
st.latex(fr"\Large\text{{FN (False Negative) = }} {FN}")



st.write("### 1. Doğruluk (Accuracy)")

st.write("**Anlam**: Tüm tahminler içinde doğru tahminlerin oranı")

st.write("**Ne zaman kullanılır**: Sınıflar dengeli olduğunda (yaklaşık eşit dağılım)")

st.write("**Örnek 1**: 100 e-postadan 90'ını doğru sınıflandırdıysanız accuracy %90'dır.")

st.write("**Örnek 2**: 1000 müşteri arasından 850'sinin kredi kartı başvurusunu doğru değerlendirdiyseniz accuracy %85'tir.")

st.write("**Dikkat**: Dengesiz veri setlerinde yanıltıcı olabilir. %95 sağlıklı, %5 hasta olan veri setinde herkesi 'sağlıklı' tahmin etsen accuracy %95 olur ama bu iyi bir model değildir.")

st.latex(
    fr"\Large\text{{Doğruluk (Accuracy):}} \quad \frac{{TP + TN}}{{TP + TN + FP + FN}} = \frac{{{TP} + {TN}}}{{{TP} + {TN} + {FP} + {FN}}} = {accuracy:.3f}"
)


st.write("### 2. Recall (Sensitivity/Duyarlılık) (True Positive Rate)")

st.write("**Anlam**: Gerçek pozitiflerin ne kadarını yakalayabildiğiniz")

st.write("**Ne zaman kullanılır**: Yanlış negatif tahminlerin maliyeti yüksek olduğunda")

st.write("**Örnek 1**: Kanser teşhisi - hasta birini sağlıklı olarak kaçırmamak kritiktir")

st.write("**Örnek 2**: Kredi kartı dolandırıcılığı tespiti - gerçek dolandırıcılık işlemlerini kaçırmamak önemlidir")

st.write("**Örnek 3**: E-posta spam filtresi - önemli e-postaları spam olarak işaretlememek kritiktir")

st.write("**Önem**: Recall, kritik durumları kaçırmamak için kritiktir. Düşük recall, önemli olayları gözden kaçırmaya ve ciddi sonuçlara neden olabilir.")

st.latex(
    fr"\Large\text{{Duyarlılık (Sensitivity/Recall/True Positive Rate):}} \quad \frac{{TP}}{{TP + FN}} = \frac{{{TP}}}{{{TP} + {FN}}} = {recall:.3f}"
)

st.write("### 3. Specificity (Özgüllük)")

st.write("**Anlam**: Gerçek negatiflerin ne kadarını doğru tahmin ettiğiniz")

st.write("**Ne zaman kullanılır**: Negatif sınıfın doğru tanımlanması önemli olduğunda")

st.latex(
    fr"\Large\text{{Özgüllük (Specificity):}} \quad \frac{{TN}}{{TN + FP}} = \frac{{{TN}}}{{{TN} + {FP}}} = {specificity:.3f}"
)

st.write("### 4. Precision (Kesinlik)")

st.write("**Anlam**: Pozitif tahmin ettikleriniz içinde gerçekten pozitif olanların oranı")

st.write("**Ne zaman kullanılır**: Yanlış pozitif tahminlerin maliyeti yüksek olduğunda")

st.write("**Örnek 1**: Spam filtresi - normal e-postaları spam olarak işaretlemek istemeyiz")

st.write("**Örnek 2**: Kredi kartı dolandırıcılığı tespiti - meşru işlemleri dolandırıcılık olarak işaretlemek müşteri deneyimini bozar")

st.write("**Örnek 3**: Tıbbi teşhislerde sağlıklı birini hasta olarak teşhis etmemek önemlidir.")

st.write("**Önem**: Precision, yanlış alarmların maliyetini minimize etmek için kritiktir. Düşük precision, gereksiz müdahalelere ve kaynak israfına neden olabilir.")


st.latex(
    fr"\Large\text{{Kesinlik (Precision):}} \quad \frac{{TP}}{{TP + FP}} = \frac{{{TP}}}{{{TP} + {FP}}} = {precision:.3f}"
)


st.write("### 5. Negatif Prediktif Değer (NPV)")

st.write("**Anlam**: Negatif tahmin ettikleriniz içinde gerçekten negatif olanların oranı")

st.write("**Ne zaman kullanılır**: Negatif tahminlerin güvenilirliği önemli olduğunda")

st.write("**Örnek 1**: Kanser taraması - sağlıklı olduğu söylenen kişilerin gerçekten sağlıklı olması önemlidir")

st.write("**Örnek 2**: İşe alım süreci - reddedilen adayların gerçekten uygun olmadığından emin olmak isteriz")

st.write("**Örnek 3**: Güvenlik taraması - güvenli olduğu belirtilen kişilerin gerçekten güvenli olması kritiktir")

st.write("**Önem**: NPV, negatif tahminlerin güvenilirliğini değerlendirmek için kritiktir. Düşük NPV, yanlış güvenlik hissi yaratabilir ve potansiyel riskleri gözden kaçırmaya neden olabilir.")

st.latex(
    fr"\Large\text{{Negatif Prediktif Değer:}} \quad \frac{{TN}}{{TN + FN}} = \frac{{{TN}}}{{{TN} + {FN}}} = {npv:.3f}"
)


st.write("### 5. F1-Score")

st.write("**Anlam**: Precision ve Recall'un harmonik ortalaması")

st.write("**Ne zaman kullanılır**: Precision ve Recall arasında denge kurmanız gerektiğinde")

st.write("**Neden harmonik ortalama**: Düşük değerleri daha fazla cezalandırır, her ikisinin de yüksek olmasını zorlar")

st.write("**Örnek 1**: Spam tespiti - hem spam e-postaları yakalamak hem de normal e-postaları yanlış işaretlememek önemlidir")

st.write("**Örnek 2**: Tıbbi teşhis - hem hastaları kaçırmamak hem de yanlış teşhis koymamak kritiktir")

st.write("**Örnek 3**: Kredi riski değerlendirmesi - hem riskli müşterileri tespit etmek hem de iyi müşterileri reddetmemek gerekir")

st.write("**Önem**: F1-Score, modelin genel performansını değerlendirmek için en yaygın kullanılan metriklerden biridir. Precision ve Recall'un ikisinin de yüksek olmasını gerektirir.")

# F1-Score hesaplama
f1_score_value = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

st.latex(
    fr"\Large\text{{F1-Score:}} \quad \frac{{1}}{{\frac{{1}}{{\text{{Precision}}}} + \frac{{1}}{{\text{{Recall}}}}}} = \frac{{1}}{{\frac{{1}}{{{precision:.3f}}} + \frac{{1}}{{{recall:.3f}}}}} = {f1_score_value:.3f}"
)
