import yfinance as yf
import os

output_folder = 'currency/currency_data'
os.makedirs(output_folder, exist_ok=True)

currency_symbols = [
    'AEDUSD=X', 'AFNUSD=X', 'ALLUSD=X', 'AMDUSD=X', 'ANGUSD=X', 
    'AOAUSD=X', 'ARSUSD=X', 'AUDUSD=X', 'AWGUSD=X', 'AZNUSD=X',
    'BAMUSD=X', 'BBDUSD=X', 'BDTUSD=X', 'BGNUSD=X', 'BHDUSD=X', 
    'BIFUSD=X', 'BMDUSD=X', 'BNDUSD=X', 'BOBUSD=X', 'BRLUSD=X', 
    'BSDUSD=X', 'BTNUSD=X', 'BWPUSD=X', 'BYNUSD=X', 'BZDUSD=X', 
    'CADUSD=X', 'CDFUSD=X', 'CHFUSD=X', 'CLPUSD=X', 'CNYUSD=X', 
    'COPUSD=X', 'CRCUSD=X', 'CUPUSD=X', 'CVEUSD=X', 'CZKUSD=X', 
    'DJFUSD=X', 'DKKUSD=X', 'DOPUSD=X', 'DZDUSD=X', 'EGPUSD=X', 
    'ERNUSD=X', 'ETBUSD=X', 'EURUSD=X', 'FJDUSD=X', 'FKPUSD=X', 
    'FOKUSD=X', 'GBPUSD=X', 'GELUSD=X', 'GGPUSD=X', 'GHSUSD=X', 
    'GIPUSD=X', 'GMDUSD=X', 'GNFUSD=X', 'GTQUSD=X', 'GYDUSD=X', 
    'HKDUSD=X', 'HNLUSD=X', 'HRKUSD=X', 'HTGUSD=X', 'HUFUSD=X', 
    'IDRUSD=X', 'ILSUSD=X', 'IMPUSD=X', 'INRUSD=X', 'IQDUSD=X', 
    'IRRUSD=X', 'ISKUSD=X', 'JEPUSD=X', 'JMDUSD=X', 'JODUSD=X', 
    'JPYUSD=X', 'KESUSD=X', 'KGSUSD=X', 'KHRUSD=X', 'KMFUSD=X', 
    'KRWUSD=X', 'KWDUSD=X', 'KYDUSD=X', 'KZTUSD=X', 'LAKUSD=X', 
    'LBPUSD=X', 'LKRUSD=X', 'LRDUSD=X', 'LSLUSD=X', 'LYDUSD=X', 
    'MADUSD=X', 'MDLUSD=X', 'MGAUSD=X', 'MKDUSD=X', 'MMKUSD=X', 
    'MNTUSD=X', 'MOPUSD=X', 'MRUUSD=X', 'MURUSD=X', 'MVRUSD=X', 
    'MWKUSD=X', 'MXNUSD=X', 'MYRUSD=X', 'MZNUSD=X', 'NADUSD=X', 
    'NGNUSD=X', 'NIOUSD=X', 'NOKUSD=X', 'NPRUSD=X', 'NZDUSD=X', 
    'OMRUSD=X', 'PABUSD=X', 'PENUSD=X', 'PGKUSD=X', 'PHPUSD=X', 
    'PKRUSD=X', 'PLNUSD=X', 'PYGUSD=X', 'QARUSD=X', 'RONUSD=X', 
    'RSDUSD=X', 'RUBUSD=X', 'RWFUSD=X', 'SARUSD=X', 'SBDUSD=X', 
    'SCRUSD=X', 'SDGUSD=X', 'SEKUSD=X', 'SGDUSD=X', 'SHPUSD=X', 
    'SLLUSD=X', 'SOSUSD=X', 'SRDUSD=X', 'SSPUSD=X', 'STNUSD=X', 
    'SVCUSD=X', 'SYPUSD=X', 'SZLUSD=X', 'THBUSD=X', 'TJSUSD=X', 
    'TMTUSD=X', 'TNDUSD=X', 'TOPUSD=X', 'TRYUSD=X', 'TTDUSD=X', 
    'TVDUSD=X', 'TWDUSD=X', 'TZSUSD=X', 'UAHUSD=X', 'UGXUSD=X', 
    'USDUSD=X', 'UYUUSD=X', 'UZSUSD=X', 'VESUSD=X', 'VNDUSD=X', 
    'VUVUSD=X', 'WSTUSD=X', 'XAFUSD=X', 'XCDUSD=X', 'XDRUSD=X', 
    'XOFUSD=X', 'XPFUSD=X', 'YERUSD=X', 'ZARUSD=X', 'ZMWUSD=X', 
    'ZWLUSD=X'

]
    # 'BIFUSD=X', 'BMDUSD=X', 'BNDUSD=X', 'BOBUSD=X', 'BRLUSD=X', 
    # 'BSDUSD=X', 'BTNUSD=X', 'BWPUSD=X', 'BYNUSD=X', 'BZDUSD=X', 
    # 'CADUSD=X', 'CDFUSD=X', 'CHFUSD=X', 'CLPUSD=X', 'CNYUSD=X', 
    # 'COPUSD=X', 'CRCUSD=X', 'CUPUSD=X', 'CVEUSD=X', 'CZKUSD=X', 
    # 'DJFUSD=X', 'DKKUSD=X', 'DOPUSD=X', 'DZDUSD=X', 'EGPUSD=X', 
    # 'ERNUSD=X', 'ETBUSD=X', 'EURUSD=X', 'FJDUSD=X', 'FKPUSD=X', 
    # 'FOKUSD=X', 'GBPUSD=X', 'GELUSD=X', 'GGPUSD=X', 'GHSUSD=X', 
    # 'GIPUSD=X', 'GMDUSD=X', 'GNFUSD=X', 'GTQUSD=X', 'GYDUSD=X', 
    # 'HKDUSD=X', 'HNLUSD=X', 'HRKUSD=X', 'HTGUSD=X', 'HUFUSD=X', 
    # 'IDRUSD=X', 'ILSUSD=X', 'IMPUSD=X', 'INRUSD=X', 'IQDUSD=X', 
    # 'IRRUSD=X', 'ISKUSD=X', 'JEPUSD=X', 'JMDUSD=X', 'JODUSD=X', 
    # 'JPYUSD=X', 'KESUSD=X', 'KGSUSD=X', 'KHRUSD=X', 'KMFUSD=X', 
    # 'KRWUSD=X', 'KWDUSD=X', 'KYDUSD=X', 'KZTUSD=X', 'LAKUSD=X', 
    # 'LBPUSD=X', 'LKRUSD=X', 'LRDUSD=X', 'LSLUSD=X', 'LYDUSD=X', 
    # 'MADUSD=X', 'MDLUSD=X', 'MGAUSD=X', 'MKDUSD=X', 'MMKUSD=X', 
    # 'MNTUSD=X', 'MOPUSD=X', 'MRUUSD=X', 'MURUSD=X', 'MVRUSD=X', 
    # 'MWKUSD=X', 'MXNUSD=X', 'MYRUSD=X', 'MZNUSD=X', 'NADUSD=X', 
    # 'NGNUSD=X', 'NIOUSD=X', 'NOKUSD=X', 'NPRUSD=X', 'NZDUSD=X', 
    # 'OMRUSD=X', 'PABUSD=X', 'PENUSD=X', 'PGKUSD=X', 'PHPUSD=X', 
    # 'PKRUSD=X', 'PLNUSD=X', 'PYGUSD=X', 'QARUSD=X', 'RONUSD=X', 
    # 'RSDUSD=X', 'RUBUSD=X', 'RWFUSD=X', 'SARUSD=X', 'SBDUSD=X', 
    # 'SCRUSD=X', 'SDGUSD=X', 'SEKUSD=X', 'SGDUSD=X', 'SHPUSD=X', 
    # 'SLLUSD=X', 'SOSUSD=X', 'SRDUSD=X', 'SSPUSD=X', 'STNUSD=X', 
    # 'SVCUSD=X', 'SYPUSD=X', 'SZLUSD=X', 'THBUSD=X', 'TJSUSD=X', 
    # 'TMTUSD=X', 'TNDUSD=X', 'TOPUSD=X', 'TRYUSD=X', 'TTDUSD=X', 
    # 'TVDUSD=X', 'TWDUSD=X', 'TZSUSD=X', 'UAHUSD=X', 'UGXUSD=X', 
    # 'USDUSD=X', 'UYUUSD=X', 'UZSUSD=X', 'VESUSD=X', 'VNDUSD=X', 
    # 'VUVUSD=X', 'WSTUSD=X', 'XAFUSD=X', 'XCDUSD=X', 'XDRUSD=X', 
    # 'XOFUSD=X', 'XPFUSD=X', 'YERUSD=X', 'ZARUSD=X', 'ZMWUSD=X', 
    # 'ZWLUSD=X'

start_date = '1994-11-29'
end_date = '2024-11-29'
for symbol in currency_symbols:
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        
        data.reset_index(inplace=True)
        clean_data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        currency_code = symbol.replace('USD=X', '')
        output_file = os.path.join(output_folder, f'{currency_code}.csv')
        clean_data.to_csv(output_file, index=False)

    except Exception as e:
        print("")
