library(Rcpp)
library(igraph)

write.csv(df, "netmix_training_data.csv", row.names = FALSE)
cat("完了: netmix_training_data.csv を保存しました。\n")
