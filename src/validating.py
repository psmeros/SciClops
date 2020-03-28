from pathlib import Path

############################### CONSTANTS ###############################

sciclops_dir = str(Path.home()) + '/data/sciclops/' 
hn_vocabulary = set(map(str.lower, open(sciclops_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()))

#Query for https://data.gesis.org/claimskg/sparql

# PREFIX schema:<http://schema.org/>
# PREFIX nif:<http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>

# SELECT ?claimText ?claimKeywords (AVG(?reviewRating) as ?AvgReviewRating)
# (GROUP_CONCAT(DISTINCT ?claimEntity;separator=",") as ?claimEntities) 

# WHERE {
# ?claim a schema:CreativeWork.
# ?claim schema:text ?claimText.
# ?claim schema:keywords ?claimKeywords.
# ?claim schema:mentions/nif:isString ?claimEntity.

# ?claimReview schema:itemReviewed ?claim.
# ?claimReview schema:reviewRating/schema:ratingValue ?reviewRating.

# } GROUP BY ?claimText ?claimKeywords



############################### ######### ###############################

import pandas as pd


pd.read_csv(sciclops_dir+'small_files/claimKG/claims.csv')