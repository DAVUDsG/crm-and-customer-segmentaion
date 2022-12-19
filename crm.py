import pandas as pd
import datetime as dt
#!pip install lifetimes

from sklearn.preprocessing import MinMaxScaler
from lifetimes import GammaGammaFitter
from lifetimes import BetaGeoFitter
df_=pd.read_csv("flo_data_20k.csv")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df_
df = df_.copy()
df.describe().T
df.isnull().sum()
df.isnull().values.any()
df.columns
df.dtypes
df
df.groupby("master_id").agg({"last_order_channel":"sum"})
df["master_id"].nunique() #tüm master_idler unique'dir
df["order_sum"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["customer_value_total_ever"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df['first_order_date'] = pd.to_datetime(df['first_order_date'])
df['last_order_date_offline '] = pd.to_datetime(df['last_order_date_offline'])
df['last_order_date'] = pd.to_datetime(df['last_order_date'])
df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])
df.groupby("master_id").agg({"order_sum":"sum","customer_value_total_ever":"sum"})
df["customer_value_total_ever"].sort_values(ascending=False).head(10)
df["order_sum"].sort_values(ascending=False).head(10)

def on_is(df):
    df["order_sum"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
    df["customer_value_total_ever"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    df['first_order_date'] = pd.to_datetime(df['first_order_date'])
    df['last_order_date_offline '] = pd.to_datetime(df['last_order_date_offline'])
    df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])
    df.groupby("master_id").agg({"order_sum": "sum", "customer_value_total_ever": "sum"})
    df["customer_value_total_ever"].sort_values(ascending=False).head(10)
    df["order_sum"].sort_values(ascending=False).head(10)
    print()

on_is(df)

df["last_order_date"].max()
today_date = dt.datetime(2021,6,1)
rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'order_sum': lambda order_sum: order_sum.nunique(),
                                     'customer_value_total_ever': lambda customer_value_total_ever: customer_value_total_ever.sum()})
rfm
rfm.columns = ["recency","frequency","monetary"]



rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.describe().T

rfm[rfm["RF_SCORE"] == "55"]

rfm[rfm["RF_SCORE"] == "11"]


# RFM isimlendirmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
#----------------------------------------------------------------------------------

rfm[(rfm["segment"]== "champions") | (rfm["segment"]=="loyal_customers")]
df.head()

set1 = rfm[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")]

set2 = df[df["interested_in_categories_12"].str.contains("KADIN")]

son_set = set1.merge(set2, on="master_id", how="left")

son_set["master_id"].index

son_set.to_csv("son_set.csv")

#---------------------------------------------------------
ts = df[df["interested_in_categories_12"].str.contains("ERKEK") | (df[df["interested_in_categories_12"].str.contains("COCUK")]
ts1 = rfm[(rfm["segment"] == "about_to_sleep") | (rfm["segment"] == "new_customers")]
cikarim = ts1.merge(ts, on="master_id", how="left")

# --------------------------------------------------------------
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    round(low_limit)
    round(up_limit)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df["customer_value_total_ever_online"].describe().T
df["customer_value_total_ever_offline"].describe().T
df["order_num_total_ever_offline"].describe().T
df["order_num_total_ever_online"].describe().T

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

df["last_order_date"].max()
today_date = dt.datetime(2021,6,1)
df.dtypes
df
#görev 2 adım2
df["customer_value_total_ever_mean"] = df["customer_value_total_ever"] / df["order_sum"]
df["recency_cltv"] = df["last_order_date"]- df["first_order_date"]
df["recency_cltv"] = df["recency_cltv"].astype('timedelta64[D]')
df["recency_cltv"] = df["recency_cltv"] / 7
df["T"] = today_date - df["first_order_date"]
df["T"] = df["T"].astype('timedelta64[D]')
df["T"] = df["T"]/7

cltv_df = df[["master_id","recency_cltv","T","order_sum","customer_value_total_ever_mean"]]

cltv_df.columns = ['customer_id','recency_cltv_weekly','T_weekly','frequency','monetary_cltv_avg']

#Görev 3 BG/NBD

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# 3 ay içerisindeki müşteri satın almalarını tahmin ediniz?
bgf.predict(12,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

bgf.predict(12,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly']).sum()
# 6 ay içerisindeki müşteri satın almalarını tahmin ediniz?
bgf.predict(24,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_6_month"] = bgf.predict(24,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

bgf.predict(24,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly']).sum()
#Gamma modelini kur
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary_cltv_avg']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary_cltv_avg']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary_cltv_avg'])
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

# 6 Aylık CLTV Hesapla DataFrame'e ekle en yüksek 20 kişiyi gözlemle
#_________________________________-

#
