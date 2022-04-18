import pandas

strFolder = 'C:\\File\\Married-Couple-Identification\\data\\'
df_transfer = pandas.read_csv(strFolder + 'havale_bil.csv')
df_statement = pandas.read_csv(strFolder + 'kk_ekstre_bil.csv')
df_transfer_statement = pandas.read_csv(strFolder + 'kk_ekstre_odm_bil.csv')
df_transactions = pandas.read_csv(strFolder + 'kk_har_bilgi.csv')
df_demog_1 = pandas.read_csv(strFolder + 'musteri_kitle_100k.csv')
df_demog_2 = pandas.read_csv(strFolder + 'sube_isl_ziy.csv')

dct_transfer = {"musteri_id_mask":"customer_id",
"tarih":"internal_transfer_date",
"alici_musteri_id_mask":"destination_id",
"doviz_fmt":"currency",
"odeme_turu_fmt":"internal_transaction_type",
"havale_tutari":"internal_transfer_amount",
"musteri_tip":"destination_type"}

dct_statement = {"musteri_id_mask":"customer_id",
"tarih":"end_of_month_date",
"hesap_id_mask":"cc_id",
"ekstre_bakiyesi_tl":"statement_amount_TL",
"ekstre_bakiyesi_usd":"statement_amount_USD",
"ekstre_bakiyesi_eur":"statement_amount_Euro",
"son_ekstre_kesim_tarihi":"statement_date",
"son_odeme_tar":"statement_due_date"}

dct_transfer_statement = {"musteri_id_mask":"customer_id",
"hesap_id_mask":"cc_id",
"islem_tarihi":"payment_date",
"doviz_fmt":"statement_currency",
"toplam_tutar":"payment_amount"}

dct_transactions = {"musteri_id_mask":"customer_id",
"islem_tarihi":"transaction_date",
"islem_saat":"transaction_time",
"islem_tutari":"transaction_total",
"mcc":"trans_spending_category",
"uyeisyeri_id_mask":"merch_id",
"harcama_tipi":"transaction_type",
"online_islem":"online_payment",
"doviz_fmt":"currency",
"x":"merch_x_coord",
"y":"merch_y_coord"}

dct_demog_1 = {"musteri_id_mask":"customer_id",
"yas":"customer_age",
"cinsiyeti":"customer_gender",
"medeni_drm_ack":"customer_marital_status",
"gelir":"customer_income_level",
"egitim_drm_ack":"customer_education_level",
"is_turu_ack":"customer_job_status",
"banka_yasi":"akbank_banking_age",
"risk_kodu_201407 -- risk_kodu_201506":"akbank_payment_action",
"mbb_segment_ack":"",
"sube_kodu":"branch_id",
"x_sube":"customer_main_branch_x_coord",
"y_sube":"customer_main_branch_y_coord",
"x_ev":"customer_home_x_coord",
"y_ev":"customer_home_y_coord",
"x_is":"customer_work_x_coord",
"y_is":"customer_work_x_coord"}

dct_demog_2 = {"musteri_id_mask":"customer_id",
"bilet_alma_tarihi":"branch_visit_date",
"min_bilet_alma_zamani":"branch_visit_time",
"sube_kodu":"branch_id",
"x":"branch_x_coord",
"y":"branch_y_coord",
"gunluk_toplam_islem_sayisi":"total_daily_transactions"}

df_transfer = df_transfer.rename(columns=dct_transfer)
df_statement = df_statement.rename(columns=dct_statement)
df_transfer_statement = df_transfer_statement.rename(columns=dct_transfer_statement)
df_transactions = df_transactions.rename(columns=dct_transactions)
df_demog_1 = df_demog_1.rename(columns=dct_demog_1)
df_demog_2 = df_demog_2.rename(columns=dct_demog_2)

print(df_transfer.columns.values)
print(df_statement.columns.values)
print(df_transfer_statement.columns.values)
print(df_transactions.columns.values)
print(df_demog_1.columns.values)
print(df_demog_2.columns.values)



