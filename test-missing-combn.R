library(data.table)
item_store_mean = fread("/media/3TB/kag/salesforecast/results/ma8dwof.csv")
item_store_family_mean_fixzeros = fread("/media/3TB/kag/salesforecast/results/ma8dwof-fixzeros.csv")
test = fread("/media/3TB/kag/salesforecast/input/test.csv")
item_store_mean = merge(item_store_mean,test,by="id")
item_store_family_mean_fixzeros = merge(item_store_family_mean_fixzeros,test,by="id")
items = fread("/media/3TB/kag/salesforecast/input/items.csv")
stores = fread("/media/3TB/kag/salesforecast/input/stores.csv")
item_store_mean = merge(item_store_mean,items,by='item_nbr',all.x=TRUE)
item_store_mean = merge(item_store_mean,stores,by='store_nbr',all.x=TRUE)
item_store_family_mean_fixzeros = merge(item_store_family_mean_fixzeros,items,by='item_nbr',all.x=TRUE)
item_store_family_mean_fixzeros = merge(item_store_family_mean_fixzeros,stores,by='store_nbr',all.x=TRUE)
combn = merge(item_store_mean[,c("id","unit_sales"),with=FALSE],item_store_family_mean_fixzeros,by="id")
missing_item_stores = combn[unit_sales.x==0,]
missing_itemfamily_stores = combn[unit_sales.y==0,]
finduniques <- function(dt,cols){
  return(unique(dt[,cols,with=FALSE]))
}
missing_itemfamily_stores$store_nbr = as.character(missing_itemfamily_stores$store_nbr)
missing_itemfamily_stores$item_nbr = as.character(missing_itemfamily_stores$item_nbr)
missing_fam_store = finduniques(missing_itemfamily_stores,c("family","store_nbr"))
missing_fam_city = finduniques(missing_itemfamily_stores,c("family","city"))
missing_item_store_unique = finduniques(missing_item_stores,c("item_nbr","store_nbr"))
