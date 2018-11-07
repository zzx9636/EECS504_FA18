from match_SIFT import *

args = parse_args()
pos_queries, num_queries = get_queries(args.query_list)
print( 'num of church to search = ' + str(num_queries))

flicker_api = FlickrAPI(flickrAPIKey, flickrSecret)

if not args.json_loc:
    json_loc = './list_json'
    for current_tag in range(0, num_queries):
        image_retrieval_json(flicker_api, pos_queries[current_tag]+' facade',json_loc)
else:
    json_loc = args.json_loc

json_list = retrive_json(json_loc) 

Histogram={}
for cur_church in json_list:
    church_name = ((cur_church.split('/')[-1]).split('.')[0]).replace('_facade','')
    print('Find match from photos of '+church_name)    
    cur_match = retrive_img(cur_church, False)
    Histogram[church_name] = cur_match
best_match = max(Histogram, key=Histogram.get)
print("Classified match is "+best_match)


