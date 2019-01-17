library(dplyr)
library(readr)
library(tibble)
library(tidyr)
library(ggplot2)
library(grid)
library(gridExtra)
library(scales)
setwd('C:/Users/Jameel shaik/Documents/Projects/Personalized Medicine Redefining Cancer Treatment/')

training_variants = read.csv("training_variants")
#########################################################
#Exploratory data analysis on training data
#Train data
Train_var = training_variants
#Checking on the data and observations
nrow(Train_var)
## [1] 3321
ncol(Train_var)
## [1] 4
#Structure of the data
str(Train_var)
## 'data.frame':    3321 obs. of  4 variables:
##  $ ID       : int  0 1 2 3 4 5 6 7 8 9 ...
##  $ Gene     : Factor w/ 264 levels "ABL1","ACVR1",..: 86 40 40 40 40 40 40 40 40 40 ...
##  $ Variation: Factor w/ 2996 levels "1_2009trunc",..: 2631 2858 1899 1668 1448 2725 2729 470 2925 213 ...
##  $ Class    : int  1 2 2 3 4 4 5 1 4 4 ...
#Data type conversion to factors
Train_var <- Train_var %>%mutate(Gene = factor(Gene),
                                 Variation = factor(Variation),
                                 Class = factor(Class))

str(Train_var)
## 'data.frame':    3321 obs. of  4 variables:
##  $ ID       : int  0 1 2 3 4 5 6 7 8 9 ...
##  $ Gene     : Factor w/ 264 levels "ABL1","ACVR1",..: 86 40 40 40 40 40 40 40 40 40 ...
##  $ Variation: Factor w/ 2996 levels "1_2009trunc",..: 2631 2858 1899 1668 1448 2725 2729 470 2925 213 ...
##  $ Class    : Factor w/ 9 levels "1","2","3","4",..: 1 2 2 3 4 4 5 1 4 4 ...
#Check for misisng values
sum(is.na(Train_var))
## [1] 0
#Since there are no missing values its gonna be easy to further analyse the data
#The proportion of all the classes 1-9 present in the data 
prop.table(table(Train_var$Class))
## 
##           1           2           3           4           5           6 
## 0.171032821 0.136103583 0.026799157 0.206564288 0.072869618 0.082806384 
##           7           8           9 
## 0.286961759 0.005721168 0.011141223
#There is about 28% of class 7

#Grouping the train data by gene
Train_gp_gen = Train_var %>%
  group_by(Gene) %>%
  summarise(cntg = n()) %>%
  arrange(desc(cntg)) %>%
  filter(cntg>20)

#Grouping the training data by variation
Train_gp_var = Train_var %>%
  group_by(Variation) %>%
  summarise(cntv = n())%>%
  arrange(desc(cntv))%>%
  filter(cntv>2)

#Plotting the classes of gene mutations 1-9
qplot(x =  Train_gp_gen$Gene, y = Train_gp_gen$cntg, data = Train_gp_gen, 
      xlab = "Gene",ylab = "Count")

#Plotting the variations in train data
qplot(x =  Train_gp_var$Variation, y = Train_gp_var$cntv, data = Train_gp_var, 
      xlab = "Variation",ylab = "Count")

#Aggregating the genes by class and plotting 
Grp_cl_gen = aggregate(Train_var$Gene, by=list(Train_var$Class),FUN= n_distinct) 
colnames(Grp_cl_gen) = c('Class','Genes')
head(Grp_cl_gen)
##   Class Genes
## 1     1   142
## 2     2    96
## 3     3    26
## 4     4    92
## 5     5    48
## 6     6    56
qplot(x =  Grp_cl_gen$Genes, y = Grp_cl_gen$Class)

#Aggregating the variation by Class and plotting
Grp_cl_var = aggregate(Train_var$Variation, by=list(Train_var$Class),FUN= n_distinct) 
colnames(Grp_cl_var) = c('Class','Variation')
head(Grp_cl_var)
##   Class Variation
## 1     1       423
## 2     2       399
## 3     3        89
## 4     4       669
## 5     5       242
## 6     6       265
qplot(x =  Grp_cl_var$Variation , y = Grp_cl_var$Class)

########################################################
library(tidytext)
library(tm)
library(stringr)
library(janeaustenr)
library(SnowballC)

#Text analysis
#Reading the textual data
train_txt_temp <- tibble(text = read_lines('training_text', skip = 1))
#Separating the columns in a structure of a dataframe
train_txt <- train_txt_temp %>%
  separate(text, into = c("ID", "txt"), sep = "\\|\\|")
#Converting the ID column to integer
train_txt <- train_txt %>%
  mutate(ID = as.integer(ID))
head(train_txt)
## # A tibble: 6 × 2
##      ID
##   <int>
## 1     0
## 2     1
## 3     2
## 4     3
## 5     4
## 6     5
## # ... with 1 more variables: txt <chr>
#Computing the length of entire text
train_txt = train_txt %>%
  mutate(txt_len = str_length(txt), 
         set ="train")

library(janeaustenr)
library(dplyr)
library(stringr)
#Spliting the sentences into single rows-per word using tidy text method
tidy_txt = train_txt %>%
  unnest_tokens(word,txt)
head(tidy_txt)
## # A tibble: 6 × 4
##      ID txt_len   set      word
##   <int>   <int> <chr>     <chr>
## 1     0   39673 train    cyclin
## 2     0   39673 train dependent
## 3     0   39673 train   kinases
## 4     0   39673 train      cdks
## 5     0   39673 train  regulate
## 6     0   39673 train         a
#Removing the stop words from the data
#Using anti-join
data(stop_words)
tidy_txt <- tidy_txt %>%
  anti_join(stop_words)%>%
  filter(str_detect(word,"[a-z]"))
## Joining, by = "word"
#Counting the total times the word occurs- Frequency
#The mutation is most frequently used in all of the documents
tidy_txt%>%
  count(word, sort=TRUE)
## # A tibble: 153,826 × 2
##         word      n
##        <chr>  <int>
## 1  mutations 237990
## 2      cells 185677
## 3       cell 126496
## 4   mutation 105444
## 5        fig 104437
## 6         al 104344
## 7     cancer  98976
## 8     figure  97894
## 9   patients  86400
## 10   protein  84289
## # ... with 153,816 more rows
head(tidy_txt)
## # A tibble: 6 × 4
##      ID txt_len   set   word
##   <int>   <int> <chr>  <chr>
## 1     0   39673 train cyclin
## 2     0   39673 train cyclin
## 3     0   39673 train cyclin
## 4     0   39673 train cyclin
## 5     0   39673 train cyclin
## 6     0   39673 train cyclin
#Plot the frequently occuring words

tidy_txt %>%
  count(word, sort = TRUE) %>%
  mutate(word = reorder(word, n)) %>%
  filter(n > 100000) %>%
  ggplot(aes(word, n)) +
  geom_density() +
  xlab(NULL) +
  coord_flip()

#Converting the words to its root form
tidy_txt <- tidy_txt %>%
  mutate(word = wordStem(word))

library(tokenizers)
library(tidyr)
#Combining the variants data with textual data for differentiating them by "Class"
Train_var%>%
  select(ID, Class)
##        ID Class
## 1       0     1
## 2       1     2
## 3       2     2
## 4       3     3
## 5       4     4
## 6       5     4
counting_t1 = full_join(Train_var,tidy_txt, by = "ID")

#Grouping the text as per the classes
frequency <- counting_t1%>%
  count(Class, word) %>%
  group_by(Class) %>%
  mutate(proportion = n / sum(n)) %>% 
  select(-n) %>% 
  spread(Class, proportion) %>% 
  gather(Class, proportion, `2`,`6`)

#Plotting class 8 frequency versus classes 2 and 6
#This shows that the words cell and express most frequently occur in all the 3 classes
#Comparing the class8 with class2 shows the word base occurs frequently in both the classes 
ggplot(frequency,aes(x=frequency$proportion, y = `8`, color = abs(`8`- frequency$proportion ))) +
  geom_abline(color = "gray80", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.1, height = 0.1) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  scale_color_gradient(limits = c(0, 0.001), low = "darkslategray4", high = "gray75") +
  facet_wrap(facets = ~ Class, ncol=2) +
  theme(legend.position="none") +
  labs(y = "Class 8", x = NULL)

#Correlation plot to see how well both the classes are correlated with class 8
cor.test(data = frequency[frequency$Class == "2",],
         ~ proportion + `8`)
## 
##  Pearson's product-moment correlation
## 
## data:  proportion and 8
## t = 140.16, df = 6096, p-value < 2.2e-16
## alternative hypothesis: true correlation is not equal to 0
## 95 percent confidence interval:
##  0.8675190 0.8794134
## sample estimates:
##       cor 
## 0.8735966
cor.test(data = frequency[frequency$Class == "6",],
         ~ proportion + `8`)
## 
##  Pearson's product-moment correlation
## 
## data:  proportion and 8
## t = 76.638, df = 4828, p-value < 2.2e-16
## alternative hypothesis: true correlation is not equal to 0
## 95 percent confidence interval:
##  0.7278435 0.7533025
## sample estimates:
##      cor 
## 0.740839
#Class 2 has a similarity of about 87%
#Tidy text mechanism for text mining
library(tidytext)
#Applying sentiments to the words
get_sentiments("afinn")
## # A tibble: 2,476 × 2
##          word score
##         <chr> <int>
## 1     abandon    -2
## 2   abandoned    -2
## 3    abandons    -2
## 4    abducted    -2
## 5   abduction    -2
## 6  abductions    -2
## 7       abhor    -3
## 8    abhorred    -3
## 9   abhorrent    -3
## 10     abhors    -3
## # ... with 2,466 more rows
get_sentiments("nrc")
## # A tibble: 13,901 × 2
##           word sentiment
##          <chr>     <chr>
## 1       abacus     trust
## 2      abandon      fear
## 3      abandon  negative
## 4      abandon   sadness
## 5    abandoned     anger
## 6    abandoned      fear
## 7    abandoned  negative
## 8    abandoned   sadness
## 9  abandonment     anger
## 10 abandonment      fear
## # ... with 13,891 more rows
get_sentiments("bing")
## # A tibble: 6,788 × 2
##           word sentiment
##          <chr>     <chr>
## 1      2-faced  negative
## 2      2-faces  negative
## 3           a+  positive
## 4     abnormal  negative
## 5      abolish  negative
## 6   abominable  negative
## 7   abominably  negative
## 8    abominate  negative
## 9  abomination  negative
## 10       abort  negative
## # ... with 6,778 more rows
# tidy_books <- counting_t1 %>%
#   group_by(Class) %>%
#   mutate(linenumber = row_number()) %>%
#   ungroup() %>%
#   unnest_tokens(word, text)
#NRC text gives emotions such as sad, anger, fear etc to the words
#Filtering the sad words from data
nrcjoy <- get_sentiments("nrc") %>% 
  filter(sentiment == "sadness")

counting_t1 %>%
  filter(Class == "8") %>%
  inner_join(nrcjoy) %>%
  count(word, sort = TRUE)
## Joining, by = "word"
## # A tibble: 57 × 2
##        word     n
##       <chr> <int>
## 1    cancer   695
## 2   inhibit   185
## 3      loss   143
## 4    tumour   122
## 5  leukemia    96
## 6   sarcoma    85
## 7     treat    81
## 8     lower    64
## 9   repress    59
## 10    death    58
## # ... with 47 more rows
#It looks like cancer is the word obvious to occcur most number of times

#The BING sentiments gives scores to the words 
janeaustensentiment <- counting_t1 %>%
  inner_join(get_sentiments("bing")) %>%
  count(Class, index = row_number() %/% 100, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)
## Joining, by = "word"
#plotting by index for different classes
ggplot(janeaustensentiment, aes(index, sentiment, fill = Class)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Class, ncol = 2, scales = "free_x")

#Looks like lots of negative words in class 4 and 5

#Comparing the 3 types for class 8
Cla_8 <- counting_t1 %>% 
  filter(Class == "8")

head(Cla_8)
##    ID  Gene Variation Class txt_len   set      word
## 1 121 SF3B1     K700R     8  135678 train    depend
## 2 121 SF3B1     K700R     8  135678 train    depend
## 3 121 SF3B1     K700R     8  135678 train    depend
## 4 121 SF3B1     K700R     8  135678 train   varieti
## 5 121 SF3B1     K700R     8  135678 train fundament
## 6 121 SF3B1     K700R     8  135678 train  cellular
#Applying all 3 sentiments for class 8
afinn <- Cla_8 %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(index = row_number() %/% 80) %>% 
  summarise(sentiment = sum(score)) %>% 
  mutate(method = "AFINN")
## Joining, by = "word"
#Converting the nrc method to positive and negative so that it is easier to plot
bing_and_nrc <- bind_rows(Cla_8 %>% 
                            inner_join(get_sentiments("bing")) %>%
                            mutate(method = "Bing et al."),
                          Cla_8 %>% 
                            inner_join(get_sentiments("nrc") %>% 
                                         filter(sentiment %in% c("positive", 
                                                                 "negative"))) %>%
                            mutate(method = "NRC")) %>%
  count(method, index = row_number() %/% 80, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)
## Joining, by = "word"
## Joining, by = "word"
bind_rows(afinn, 
          bing_and_nrc) %>%
  ggplot(aes(index, sentiment, fill = method)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~method, ncol = 1, scales = "free_y")

#Since the AFINN and Bing methods are binary we have both extremes
#Whereas the NRC has various scores in the plot- this shows how sentiments are
#assigned to individual words and plotted against their scores.

#Assigns positives and negatives to each word and computed their frequency
bing_word_counts <- tidy_txt %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()
## Joining, by = "word"
head(bing_word_counts)
## # A tibble: 6 × 3
##      word sentiment      n
##     <chr>     <chr>  <int>
## 1  cancer  negative 120137
## 2 patient  positive 110230
## 3    wild  negative  56917
## 4 inhibit  negative  38662
## 5 complex  negative  27566
## 6    loss  negative  24794
#Plotting them
#Progress and patient are positive words
#inhibit is a negative word as for ex the cancerous cells inhibits 
#the growth of healthy cells
bing_word_counts %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to sentiment",
       x = NULL) +
  coord_flip()
## Selecting by n

###################################################
library(wordcloud)
#Building the word cloud as per the frequency of occurance
tidy_txt %>%
  anti_join(stop_words) %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 500))
## Joining, by = "word"

#Mutate is the most frequenctly occuring word.

library(reshape2)
#Plotting the positives and negatives
tidy_txt %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("#F8766D", "#00BFC4"),
                   max.words = 200)
