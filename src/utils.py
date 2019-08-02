from urllib.parse import urlsplit
import re


#Find the domain and the path of an http url
def analyze_url(url):
    try:
        url=urlsplit(url)
        domain = re.sub(r'^(http(s)?://)?(www\.)?', r'', url.netloc)
        path = '' if domain == '' else url.path
        return domain, path
    except:
        return url, ''

# DEPRECATED
# def apriori_clustering():
#   te = TransactionEncoder()
#   te_ary = te.fit_transform(articles['refs'])

#   clusters = apriori(pd.DataFrame(te_ary, columns=te.columns_), min_support=2/len(articles), use_colnames=True, low_memory=True).rename(columns={'itemsets':'refs'})

#   def get_articles_for_cluster(cluster_refs):
#     return articles[[cluster_refs.issubset(r) for r in articles.refs]].index.tolist()

#   clusters['articles'] = clusters.refs.apply(get_articles_for_cluster)
#   clusters['length'] = clusters.articles.apply(len)
