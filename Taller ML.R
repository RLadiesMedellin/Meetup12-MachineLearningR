# LIBRERIAS ---------------------------------------------------------------
library(tidyverse)
library(gridExtra)
library(ggcorrplot)
library(readr)

# EXPLORACION DE LA BASE --------------------------------------------------

# importar la base de datos
dt <- read.csv2("datos_ibm.csv")

# 0 active employee, 1 former employee

# variables y tipo de variables
str(dt)

# valores faltantes
apply(X = is.na(dt), MARGIN = 2, FUN = sum)
sum(is.na(dt))

# cambio nombre
colnames(dt)[1] <- "Age"

# Eliminar columnas
dt <- dt %>% select(-EmployeeNumber, -StandardHours, -Over18) %>%
  mutate_if(is.character,as.factor)

dt$Education <- as.factor(dt$Education)
dt$EnvironmentSatisfaction <- as.factor(dt$EnvironmentSatisfaction)
dt$JobInvolvement <- as.factor(dt$JobInvolvement)
dt$JobSatisfaction <- as.factor(dt$JobSatisfaction)
dt$PerformanceRating <- as.factor(dt$PerformanceRating)
dt$RelationshipSatisfaction <- as.factor(dt$RelationshipSatisfaction)
dt$WorkLifeBalance <- as.factor(dt$WorkLifeBalance)



# histogramas de algunas variables numericas (edad, salario, años en la compañia, 
# años en el puesto actual, años que lleva trabjando)

par(mfrow=c(2,2))
# histograma para la edad
g1 <- dt %>% 
  ggplot(aes(x=Age)) +
  geom_histogram( binwidth=3, fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  ggtitle("Histograma de la edad")

# histograma del salario
g2 <- dt %>% 
  ggplot(aes(x=MonthlyIncome)) +
  geom_histogram(bins=30, fill="#69b3a2") +
  ggtitle("Histograma del salario")

# histograma de los años en la compañia
g3 <- dt %>% 
  ggplot(aes(x=YearsAtCompany)) +
  geom_histogram(bins=30, fill="#69b3a2") +
  ggtitle("Histograma de años en la compañia")

# histograma de años total trabajando 
g4 <- dt %>% 
  ggplot(aes(x=TotalWorkingYears)) +
  geom_histogram(bins=30, fill="#69b3a2") +
  ggtitle("Histograma de años trabajando")

grid.arrange(g1, g2, g3, g4, nrow = 2)

# agrupar por la variable respuesta (reunucio o no) y visualizar las otras variables
# edad, estado civil, salario, años en la compania, GENERO)

# conteo empleados vs ex empleados
dt %>% 
  group_by(Resignation) %>% 
  summarise(conteo = n())

# histograma retiro por edad
dt %>%
  ggplot(aes(x=Age, fill=Resignation)) +
  geom_histogram(bins=30,color="#e9ecef", alpha=0.6, position = 'identity') +
  scale_fill_manual(values=c("#69b3a2", "#404080"))  +
  ggtitle("Histograma Edad por Retiro")

# histograma genero

# crear tabla que haga conteo para hacer barplot genero
tabla <- dt %>%  
  group_by(Gender, Resignation) %>% 
  count()

ggplot(tabla, aes(x=Gender, y=n, fill=Resignation)) + 
  geom_bar(stat = "identity")

# histograma para estado civil
tabla_1 <- dt %>%  
  group_by(MaritalStatus, Resignation) %>% 
  count()

ggplot(tabla_1, aes(x=MaritalStatus, y=n, fill=Resignation)) + 
  geom_bar(stat = "identity")

# gráfico de correlación
# variables numericas
num <- dt %>% 
  select(Age, DistanceFromHome, MonthlyIncome, TotalWorkingYears,
         NumCompaniesWorked, YearsAtCompany, YearsInCurrentRole,
         YearsSinceLastPromotion, YearsWithCurrManager)

# matriz de correlación
corr <- round(cor(num), 2)

# visualizar la matriz de correlación
ggcorrplot(corr)

# MODELOS DE MACHINE LEARNING ---------------------------------------------

# Preprocesamiento de base de datos ---------------------------------------

# Split en train y test
library(caret)
library(tidymodels)
set.seed(1234)
train_test_split <- dt %>% initial_split(prop = 0.80)

train_data <- training(train_test_split)
test_data <- testing(train_test_split)


# Imbalanceo de clases
train_data %>% group_by(Resignation) %>% count()

# Oversampling
library(ROSE)
train_balanced <- ovun.sample(Resignation ~ ., data = train_data, method = "over")$data
train_balanced %>% group_by(Resignation) %>% count()

preprocesamiento <- function(datos) {
  
  # Variables categoricas a dummies
  cat_datos <- datos %>% select_if(is.factor)
  cat_dummies <- dummyVars("~.", data = cat_datos, fullRank = TRUE)
  dat_transformed <- data.frame(predict(cat_dummies, newdata = cat_datos))
  
  # Escalado de variables numéricas
  num_datos <- datos %>% select_if(is.numeric)
  num_scale <- scale(num_datos)
  
  # Unir ambas bases de datos
  datos_final <- cbind(dat_transformed, num_scale)
  
  return(datos_final)
}

# base de datos de entrenamiento final
train_final <- preprocesamiento(train_balanced)
colnames(train_final)[1] <- "Target"
train_final$Target <- as.factor(train_final$Target)

# Entrenamiento de modelos de clasificación -------------------------------

# Regresión logística
fit_log <- glm(Target ~., data = train_final, family = "binomial")
summary(fit_log)

# Árboles de decisión
library(rpart)
library(rpart.plot)
fit_tree <- rpart(Target~., data = train_final, method = 'class')
rpart.plot(fit_tree)

# Bosque aleatorio
library(randomForest)
set.seed(1234)  # Fijar semilla
fit_forest <- randomForest(x = select(train_final, -Target),
                           y = train_final$Target,
                           ntree = 100)

fit_forest
# importancia de variables
varImpPlot(fit_forest)



# Evaluar desempeño de los modelos ----------------------------------------

# preprocesamiento de datos de prueba
test_final <- preprocesamiento(test_data)
colnames(test_final)[1] <- "Target"
test_final$Target <- as.factor(test_final$Target)

# Predicciones Regresión logística
predict_log <- predict(fit_log, 
                       test_final, 
                       type = "response")
predict_log_group <- ifelse(predict_log >0.5, 1, 0)
predict_log_group <- as.factor(predict_log_group)
# Matriz de confusión
confusionMatrix(predict_log_group, test_final$Target)

# Predicciones árbol de decisión
predict_tree <- predict(fit_tree, 
                        test_final, 
                        type = "class")
# Matriz de confusión
confusionMatrix(predict_tree, test_final$Target)


# Predicciones bosque aleatorio
predict_forest <- predict(fit_forest, 
                          test_final, 
                          type = "class")
# Matriz de confusión
confusionMatrix(predict_forest, test_final$Target)
