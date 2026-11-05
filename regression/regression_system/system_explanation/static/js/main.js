document.addEventListener('DOMContentLoaded', () => {
    // 1. TAB SEÇİMİ VE EKRAN DEĞİŞTİRME MANTIĞI
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Tüm menü elemanlarının aktifliğini kaldır
            navItems.forEach(nav => nav.classList.remove('active'));
            // Tıklananı aktif yap
            item.classList.add('active');

            // Tüm sekme içeriklerini gizle
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Tıklanana ait sekmeyi bul ve göster
            const targetTabId = item.getAttribute('data-tab');
            const targetTab = document.getElementById(targetTabId);
            if (targetTab) {
                targetTab.classList.add('active');
            }
        });
    });

    // 2. TAHMİN FORMU GÖNDERİM VE API ÇAĞRISI MANTIĞI
    const hba1cForm = document.getElementById('hba1c-form');
    const resultPlaceholder = document.querySelector('.result-placeholder');
    const resultDisplay = document.querySelector('.result-display');
    
    const resultVal = document.getElementById('result-val');
    const resultCircle = document.getElementById('result-circle');
    const riskStatus = document.getElementById('risk-status');
    const riskDesc = document.getElementById('risk-desc');
    
    // Hesaplanan Klinik İndeksler
    const compTgHdl = document.getElementById('comp-tg-hdl');
    const compLdlHdl = document.getElementById('comp-ldl-hdl');
    const compAksTks = document.getElementById('comp-aks-tks');
    const compGlisemik = document.getElementById('comp-glisemik');

    hba1cForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Hata alanlarını temizle
        document.querySelectorAll('.field-error').forEach(err => err.textContent = '');
        
        // Form verilerini al
        const formData = {
            yas: parseFloat(document.getElementById('yas').value),
            cinsiyet: document.getElementById('cinsiyet').value,
            aks: parseFloat(document.getElementById('aks').value),
            tks_1: parseFloat(document.getElementById('tks_1').value),
            tks_2: parseFloat(document.getElementById('tks_2').value),
            kolesterol: parseFloat(document.getElementById('kolesterol').value),
            hdl: parseFloat(document.getElementById('hdl').value),
            ldl: parseFloat(document.getElementById('ldl').value),
            tg: parseFloat(document.getElementById('tg').value)
        };

        // Ön Yüz Temel Doğrulama
        let hasError = false;
        
        // Cinsiyet Kontrolü
        if (!['E', 'K'].includes(formData.cinsiyet)) {
            showFieldError('cinsiyet', "Cinsiyet 'E' (Erkek) veya 'K' (Kadın) olmalıdır.");
            hasError = true;
        }

        // Yaş Kontrolü
        if (isNaN(formData.yas) || formData.yas < 0 || formData.yas > 120) {
            showFieldError('yas', "Geçerli bir yaş giriniz (0-120).");
            hasError = true;
        }

        // Diğer Değerlerin Negatif Olmama Kontrolü
        const numericFields = ['aks', 'tks_1', 'tks_2', 'kolesterol', 'hdl', 'ldl', 'tg'];
        numericFields.forEach(field => {
            if (isNaN(formData[field]) || formData[field] <= 0) {
                showFieldError(field, "Değer 0'dan büyük olmalıdır.");
                hasError = true;
            }
        });

        if (hasError) return;

        // Buton durumunu güncelle (Loading)
        const btnSubmit = document.querySelector('.btn-submit');
        const originalBtnText = btnSubmit.innerHTML;
        btnSubmit.disabled = true;
        btnSubmit.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Hesaplama Yapılıyor...';

        try {
            // API Tahmin Endpoint Çağrısı
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.get_json ? await response.get_json() : await response.json();

            if (response.ok) {
                // Sonuç alanını göster
                resultPlaceholder.classList.add('hidden');
                resultDisplay.classList.remove('hidden');

                // HbA1c Değerini Yaz
                resultVal.textContent = result.predicted_hba1c.toFixed(2);
                
                // Circle rengini ve gölgesini güncelle
                resultCircle.style.borderColor = result.risk_color;
                resultCircle.style.boxShadow = `0 0 25px ${result.risk_color}80`;

                // Risk Durumu ve Açıklamaları Yaz
                riskStatus.textContent = result.risk_status;
                riskStatus.style.color = result.risk_color;
                
                riskDesc.textContent = result.description;
                document.querySelector('.clinical-description').style.borderColor = result.risk_color;

                // Hesaplanan Klinik İndeksleri Yaz
                const tg_hdl_calc = formData.tg / formData.hdl;
                const ldl_hdl_calc = formData.ldl / formData.hdl;
                const aks_tks_calc = formData.aks / formData.tks_2;
                const glisemik_calc = (formData.aks + formData.tks_1 + formData.tks_2) / 3.0;

                compTgHdl.textContent = tg_hdl_calc.toFixed(2);
                compLdlHdl.textContent = ldl_hdl_calc.toFixed(2);
                compAksTks.textContent = aks_tks_calc.toFixed(2);
                compGlisemik.textContent = glisemik_calc.toFixed(2);

                // Sonuç alanına yumuşak geçişle odaklan
                document.getElementById('result-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            } else {
                alert(`Doğrulama Hatası: ${result.error}`);
            }

        } catch (error) {
            console.error('API Error:', error);
            alert('Sunucuyla bağlantı kurulamadı. Lütfen Flask sunucusunun çalıştığından emin olun.');
        } finally {
            // Butonu eski haline getir
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = originalBtnText;
        }
    });

    // Yardımcı hata gösterme fonksiyonu
    function showFieldError(fieldId, message) {
        const field = document.getElementById(fieldId);
        if (field) {
            const errorSpan = field.nextElementSibling;
            if (errorSpan && errorSpan.classList.contains('field-error')) {
                errorSpan.textContent = message;
            }
        }
    }
});
