
##################################################################
########   SCRIPT POUR RE-EXECUTER LE MODELE               #######
################ MÉTHODOLOGIE: GRADIENT BOOSTING  ################

# Il faut déclarer les routes de travail
PATH <- 'C:/Users/ach/Desktop/Fraude/'


DATADIR    <- paste(PATH, 'data/',   sep = '')
INPUTDIR   <- paste(PATH, 'input/',  sep = '')
OUTPUTDIR  <- paste(PATH, 'output/', sep = '')
SYNTAXDIR  <- paste(PATH, 'syntax/', sep = '')

library(xgboost)
library(caret)
library(pdp)

# Nous chargons l'échantillon (nouvelle échantillon avec laquelle nous fairons un nouveau modèle)
file <- paste(DATADIR, "bbdd.RData", sep = '')
aux <- load(file); cat('Archive:', file, '\n')
bbdd <- get(aux); if (!'bbdd' %in% aux) {rm(list = aux)}


# Nous chargons les variables qui sont candidates pour entrer dans le modèle
file.vars <- paste(INPUTDIR, "201807_Selection_Variables.csv", sep = '')
inf.vars <- read.table(file.vars, header = T, sep = ";", encoding = "UTF-8")
Variables <- as.character(inf.vars$VAR)


# Les variables qui ne sont pas numériques ne peuvent pas entrer dans le modèle.
# Si il y a variables categoriques nous devons les transformer en variables binaires.

# Ici, nous indiquons les variables qui ne sont pas numériques
var.char <- c("TypeVehicule", "ville_BQ", "Banque", "CodeProduit", "Agence", "Nationalite", "SituationFamiliale", "Profession",
              "TypeLogement", "TYPE_PM", "TYPE_CLIENT", "Anc_Client", "Sexe", "MarqueVehicule", "SecteurActivite","ConcessionnaireAuto")


# Nous transormons ces variables en variables binaires
dmy <- dummyVars(" ~ .", data = bbdd[, var.char], sep = ".")
dummy <- data.frame(predict(dmy, newdata = bbdd[, var.char]))
fac.dum <- colnames(dummy)[2:dim(dummy)[2]]
bbdd[, fac.dum] <- dummy[, 2:dim(dummy)[2]]

View(fac.dum)
write.table(fac.dum, file = paste(OUTPUTDIR, "VariablesBinaires.csv"), row.names = FALSE)
# nouveau fichier avec toutes ces nouveaux variables binaires

# Après le nouveau ficher a ete actualise, nous le chargons
file.vars <- paste(INPUTDIR, "201807_Selection_Variables_V2.csv", sep = '')
inf.vars <- read.table(file.vars, header = T, sep = ";", encoding = "UTF-8")
Variables <- as.character(inf.vars$VAR)
Variables.MONOT <- as.numeric(inf.vars$TENDANCE)

# Nous preparons l'echantillon pour reexecuter le modèle
matrix_bbdd <- xgb.DMatrix(as.matrix(bbdd[, Variables]), label = bbdd[["FRAUDE"]], missing = NA)

# Il y a un probleme de codification avec ces variables, nous devons les eliminer
Variables <- Variables[Variables %in% colnames(bbdd)]
matrix_bbdd <- xgb.DMatrix(as.matrix(bbdd[, Variables]), label = bbdd[["FRAUDE"]], missing = NA)


# Premier pas: Trouber le numéro optimale d'arbres à faire dans le modèle finale
# Nous définissons les paramètres pour éxecuter le modèle
seed.xgboost <- 123
par.nfold <- 5 
parametres <- list(booster = "gbtree",
                   objective = "binary:logistic",
                   eval_metric = "logloss",
                   eta = 0.03,
                   gamma = 5,
                   subsample = 1,
                   colsample_bytree = 1,
                   max_depth = 4,
                   nthread = 5)

(par.nrounds <- 1000)
(par.early_stopping_rounds <- 20)
(par.weight = NULL)
(par.maximize = FALSE)
(par.eval.metric = "logloss")

set.seed(seed.xgboost)
xgbcv <- xgb.cv(params = parametres, data = matrix_bbdd, nrounds = par.nrounds,
                nfold = par.nfold, monotone_constraints = Variables.MONOT, prediction = TRUE, metrics = par.eval.metric, 
                feval = NULL, print_every_n = 1L, early_stopping_rounds = par.early_stopping_rounds, 
                maximize = NULL)


# Numero d'arbres dans le modèle finale
numarbres <- xgbcv$best_ntreelimit

# Pouvoir discrimintacion du modèle en validation
pred <- ROCR::prediction(xgbcv$pred, bbdd$FRAUDE)
auc_temp <- ROCR::performance(pred, "auc")
auc <- as.numeric(auc_temp@y.values)
cat("Gini cross-validation ", par.nfold, "iterations:", paste(round((auc*2 - 1)*100, 2), "%", sep = ""), "\n")



# Nous définissons les paramètres pour éxecuter le modèle
parametres <- list(booster = "gbtree",
                   objective = "binary:logistic",
                   eval_metric = "logloss",
                   eta = 0.03,
                   gamma = 5,
                   subsample = 1,
                   colsample_bytree = 1,
                   max_depth = 4,
                   nthread = 5)

(par.nrounds <- numarbres)
(par.weight = NULL)
(par.maximize = FALSE)
(par.eval.metric = "logloss")

watchlist <- list(train = matrix_bbdd)
set.seed(seed.xgboost)
boost.xgboost <- xgb.train(params  = parametres,
                           monotone_constraints = Variables.MONOT,
                           data = matrix_bbdd,
                           nrounds = numarbres,
                           watchlist = watchlist, maximize = par.maximize)


# Pour evaluer l'echantillon avec le modèle et obtenir la probabilite de fraude
bbdd$PROB_FRAUDE <-predict(boost.xgboost, newdata=matrix_bbdd, ntreelimit = numarbres, outputmargin = FALSE)


# Pour voir le poivoir de prédiction du modèle
pred <- ROCR::prediction(bbdd$PROB_FRAUDE , bbdd$FRAUDE)
auc_temp <- ROCR::performance(pred, "auc")
auc <- as.numeric(auc_temp@y.values)

cat("Gini en construction:", paste(round((auc*2 - 1)*100, 2), "%", sep = ""), "\n")
#89.4

##Seuils
predxgb3 <-factor(ifelse(bbdd$PROB_FRAUDE <= 0.19 ,0 , ifelse(bbdd$PROB_FRAUDE > 0.19 & bbdd$PROB_FRAUDE <= 0.49 ,1 , 2)))
library(xlsx)
sortie1 <- data.frame(predxgb3)
sortie2<- data.frame(bbdd$PROB_FRAUDE)
sortie <- cbind(bbdd,sortie2, sortie1)
write.csv2(sortie, file="C:/Users/ach/Desktop/test.csv", na="")

#Variables du modele
imp.var <- xgb.importance(feature_names = Variables, model = boost.xgboost)


#Tendance des variables du modèle
table_tendance <- partial(boost.xgboost, 
                     pred.var = "Anc_CB", 
                     train = bbdd[, Variables], 
                     plot = FALSE,
                     rug = FALSE, 
                     prob = TRUE,
                     trim.outliers = TRUE)


table_tendance <- partial(boost.xgboost, 
                          pred.var = "ProfessionAutre", 
                          train = bbdd[, Variables], 
                          plot = FALSE,
                          rug = FALSE, 
                          prob = TRUE,
                          trim.outliers = FALSE)

View(table_tendance)

save(boost.xgboost,file = paste(OUTPUTDIR, "Modele_fraude_201809.Rdata", sep=""))



t <- quantile(bbdd$PROB_FRAUDE, probs = seq(0,1,0.1))
t

sel <- which(bbdd$PROB_FRAUDE <= 0.3)## seuil à déterminer
table(bbdd$FRAUDE[sel])

table(bbdd$FRAUDE[-sel])