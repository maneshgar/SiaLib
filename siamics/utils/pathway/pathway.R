suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)   
  library(enrichplot)
  library(ggplot2)
  library(enrichplot)
  library(ggraph)
  library(optparse)
})

option_list <- list(
  make_option(c("-o", "--output_dir"), type = "character", default = NULL,
              help = "Output directory", metavar = "character"),
  make_option(c("-g", "--gene_file"), type = "character", default = NULL,
              help = "Path to gene file (.tsv)", metavar = "character"),
  make_option(c("-t", "--title"), type = "character", default = NULL,
              help = "Plot title", metavar = "character")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Check required arguments
if (is.null(opt$output_dir) || is.null(opt$gene_file) || is.null(opt$title)) {
  print_help(opt_parser)
  stop("All three arguments must be provided.", call. = FALSE)
}

cat("Output directory:", opt$output_dir, "\n")
cat("Gene file:", opt$gene_file, "\n")
cat("Title:", opt$title, "\n")


output_dir <- opt$output_dir
gene_file  <- opt$gene_file
title <- opt$title

cat("Output directory:", output_dir, "\n")
cat("Title:", title, "\n")

# Create output dir
timestamp <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
final_dir <- file.path(output_dir, title, timestamp)
if (!dir.exists(final_dir)) {
  dir.create(final_dir, recursive = TRUE)
}

print(final_dir)

###### Process Genes
genes_df <- read.table(gene_file, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
sig_symbols <- unique(na.omit(genes_df$gene))


# gene name -> ENTREZID
conv_sig <- bitr(sig_symbols, fromType="SYMBOL", toType="ENTREZID", OrgDb=org.Hs.eg.db)
gene_entrez <- unique(conv_sig$ENTREZID)

###### GO
ego_bp <- enrichGO(gene_entrez, OrgDb=org.Hs.eg.db, keyType="ENTREZID", ont="BP",
                   pAdjustMethod="BH", pvalueCutoff=0.05, qvalueCutoff=0.2
                  )

ego_cc <- enrichGO(gene_entrez, OrgDb=org.Hs.eg.db, keyType="ENTREZID", ont="CC",
                   pAdjustMethod="BH", pvalueCutoff=0.05, qvalueCutoff=0.2
                  )

ego_mf <- enrichGO(gene_entrez, OrgDb=org.Hs.eg.db, keyType="ENTREZID", ont="MF",
                   pAdjustMethod="BH", pvalueCutoff=0.05, qvalueCutoff=0.2
                  )

# cat("Top Biological Process pathways:\n")
# print(head(arrange(as.data.frame(ego_bp), p.adjust), 5))
# 
# cat("\nTop Cellular Component pathways:\n")
# print(head(arrange(as.data.frame(ego_cc), p.adjust), 5))
# 
# cat("\nTop Molecular Function pathways:\n")
# print(head(arrange(as.data.frame(ego_mf), p.adjust), 5))

ego_bp <- setReadable(ego_bp, OrgDb=org.Hs.eg.db, keyType="ENTREZID")
ego_cc <- setReadable(ego_cc, OrgDb=org.Hs.eg.db, keyType="ENTREZID")
ego_mf <- setReadable(ego_mf, OrgDb=org.Hs.eg.db, keyType="ENTREZID")

write.csv(as.data.frame(ego_bp),file.path(final_dir, "ego_bp.csv"),row.names = FALSE)
write.csv(as.data.frame(ego_cc),file.path(final_dir, "ego_cc.csv"),row.names = FALSE)
write.csv(as.data.frame(ego_mf),file.path(final_dir, "ego_mf.csv"), row.names = FALSE)

shrink_goplot_text <- function(x, txt_size = 2.5, geom = "text") {
  p <- goplot(x, geom = geom)
  for (i in seq_along(p$layers)) {
    cls <- class(p$layers[[i]]$geom)[1]
    if (grepl("GeomNode(Text|Label)", cls) || grepl("Repel", cls)) {
      p$layers[[i]]$aes_params$size  <- txt_size
      p$layers[[i]]$geom_params$size <- txt_size
    }
  }
  p
}

p <- shrink_goplot_text(ego_bp, txt_size = 2)
ggsave(file.path(final_dir, "ego_bp_plot.png"), plot = p, width = 8, height = 6, dpi = 300)

p <- shrink_goplot_text(ego_cc, txt_size = 2)
ggsave(file.path(final_dir, "ego_cc_plot.png"), plot = p, width = 8, height = 6, dpi = 300)

p <- shrink_goplot_text(ego_mf, txt_size = 2)
ggsave(file.path(final_dir, "ego_mf_plot.png"), plot = p, width = 8, height = 6, dpi = 300)

p <- dotplot(ego_bp, showCategory = 15) + ggtitle("GO: Biological Process")
ggsave(file.path(final_dir, "dotplot_ego_bp.png"), plot = p, width = 8, height = 6, dpi = 300)

p <- dotplot(ego_cc, showCategory = 15) + ggtitle("GO: Cellular Component")
ggsave(file.path(final_dir, "dotplot_ego_cc.png"), plot = p, width = 8, height = 6, dpi = 300)


p <- dotplot(ego_mf, showCategory = 15) + ggtitle("GO: Molecular Function")
ggsave(file.path(final_dir, "dotplot_ego_mf.png"), plot = p, width = 8, height = 6, dpi = 300)


print("Finished GO Analysis")

###### KEGG

ekegg <- enrichKEGG(gene=gene_entrez, organism="hsa", pAdjustMethod="BH", pvalueCutoff=0.05, qvalueCutoff  = 0.2)

write.csv(as.data.frame(ekegg),file.path(final_dir, "ekegg.csv"),row.names = FALSE)

p <- dotplot(ekegg, showCategory = 15) + ggtitle("KEGG pathways")
ggsave(file.path(final_dir, "dotplot_ekegg.png"), plot = p, width = 8, height = 6, dpi = 300)

print("Finished KEGG Analysis")

cat("Results are saved in", final_dir, "\n")



