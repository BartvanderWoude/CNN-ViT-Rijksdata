library(xml2);
library(tidyverse);
library(stringi);
library(stringr);

#################### Creating dataframe from XML data ####################

# Import data and read it as a list
collection_xml = as_list(read_xml("collection.xml"))

# Create a tibble of the data in order for easier manipulation
data = tibble::as_tibble(collection_xml);

#clean up
rm(collection_xml)

# Create columns for all the data ids, essentially creating the data frame
data = data %>% unnest_longer(`OAI-PMH`);
data = data %>% unnest_wider(`OAI-PMH`);
data = data %>% unnest_wider(header);
data = data %>% unnest_longer(metadata);
data = data %>% unnest_wider(metadata);

# Convert data from list to actual data
data = data %>% unnest(cols = names(.)) %>% unnest(cols = names(.)) %>% readr::type_convert();
data_backup <- data;

data <- as.data.frame(data)

rm(collection_xml);
gc();

#################### Data clean-up ####################
# Exception removal
data = data[, !(names(data) %in% "datestamp")]

# Create lists of column names that are useful
identifier_names = ifelse(grepl("identifier", names(data)), TRUE, FALSE);
identifier_names = names(data)[identifier_names];

date_names = ifelse(grepl("date", names(data)), TRUE, FALSE);
date_names = names(data)[date_names];

format_names = ifelse(grepl("format", names(data)), TRUE, FALSE);
format_names = names(data)[format_names];

# Remove columns from data that aren't used
data = data[, (names(data) %in% c(identifier_names, date_names, format_names) )];

format_names = ifelse(grepl("format", names(data)), TRUE, FALSE);
format_names = names(data)[format_names];

for (r in 1:nrow(data))
{
  for (c in format_names)
  {
    if (!is.na(data[r, c]) && data[r, c] == "papier")
    {
      data[r, c] = NA
      break
    }
  }
}

# Clean up by creating full columns rather than storing data in random columns
for (r in 1:nrow(data))
{
  i = 1;
  j = 1;
  for (name in date_names)
  {
    if ( !is.na(data[r, name]) )
    {
      new_name = paste("date", toString(i), sep = "");
      data[r, new_name] = data[r, name];
      i = i + 1;
    }
  }
  
  for (name in format_names)
  {
    if ( !is.na(data[r, name]) )
    {
      new_name = paste("format", toString(j), sep = "");
      data[r, new_name] = data[r, name];
      j = j + 1;
    }
  }
}

rm(list = c("i", "j", "name", "new_name", "r"));

# Create a specific dataframe for both the dates and formats
data = data[, !(names(data) %in% c(date_names, format_names))];

date_names = ifelse(grepl("date", names(data)), TRUE, FALSE);
date_names = names(data)[date_names];

format_names = ifelse(grepl("format", names(data)), TRUE, FALSE);
format_names = names(data)[format_names];

date_data = data[, (names(data) %in% c(identifier_names, date_names))];
format_data = data[, (names(data) %in% c(identifier_names, format_names))];

rm(data);

# Remove outliers from date data and clean up the table again
date_data = date_data[!date_data$identifier...2 == "BK-KOG-2546",];

remove_names = vector();
for (name in date_names)
{
  if ( sum(!is.na(date_data[,name])) < 1 )
  {
    remove_names = c(remove_names, name);
  }
}
date_data = date_data[, !(names(date_data) %in% remove_names)];

date_names = ifelse(grepl("date", names(date_data)), TRUE, FALSE);
date_names = names(date_data)[date_names];

# Remove NA from date
remove_idx = vector()
for (r in 1:nrow(date_data))
{
  if (is.na(date_data[r, "date1"]))
  {
    remove_idx = c(remove_idx, r)
  }
}
date_data <- date_data[-remove_idx,]

# Remove outliers from format data and clean up the table again
format_data = format_data[!format_data$identifier...2 == "BK-16676",];
format_data = format_data[!format_data$identifier...2 == "BK-NM-7325",];
format_data = format_data[!format_data$identifier...2 == "BK-NM-1010",];

for (r in 1:nrow(format_data))
{
  for (c in names(format_data))
  {
    if (!is.na(format_data[r, c]) && format_data[r, c] == "papier")
    {
      format_data[r, c] = NA
      break
    }
  }
}

remove_names = vector();
for (name in format_names)
{
  if ( sum(!is.na(format_data[,name])) < 1 )
  {
    remove_names = c(remove_names, name);
  }
}
format_data = format_data[, !(names(format_data) %in% remove_names)];

format_names = ifelse(grepl("format", names(format_data)), TRUE, FALSE);
format_names = names(format_data)[format_names];

rm(list = c("name", "remove_names"));



#################### Data analysis: date ####################
# Get dates from data
for (name in date_names)
{
  first_name = paste(name, "_first", sep = "");
  last_name = paste(name, "_last", sep = "");
  
  date_data[,first_name] <- strtoi(stri_extract_first_regex(date_data[,name], '[-]*[0-9]+'));
  date_data[,last_name] <- strtoi(stri_extract_last_regex(date_data[,name], '[-]*[0-9]+'));
}

rm(list = c("last_name", "first_name","name"))

# Remove dates older than 0 AD or dates in the future (i.e. > 2022)
remove_idx = vector()
for (r in 1:nrow(date_data))
{
  if (date_data[r, "date1_first"] < 0)
  {
    remove_idx = c(remove_idx, r);
  }
  
  
  if (date_data[r, "date1_first"] > 2022)
  {
    remove_idx = c(remove_idx, r);
  }
}
date_data <- date_data[-remove_idx,]

# Assign century classes
for (r in 1:nrow(date_data))
{
  century_min = 0;
  century_max = 99;
  century = 1;
  while (century_min < 2100)
  {
    if (date_data[r, "date1_first"] >= century_min && date_data[r, "date1_first"] <= century_max)
    {
      date_data[r, "century"] = century;
      break
    }
    
    century_min = century_min + 100;
    century_max = century_max + 100;
    century = century + 1;
  }
}

rm(list = c("r", "remove_idx", "century_min", "century_max", "century"))


#################### Export data as python API handler readable file ####################
export_data <- date_data[,c("identifier...2", "century")]
names(export_data)[names(export_data) == "identifier...2"] <- "id"
write.csv(export_data, "C:/Users/beste/Documents/RUG/BachelorThesis/DataSpec/id_list_date.csv", row.names = FALSE)

format_names = ifelse(grepl("format", names(format_data)), TRUE, FALSE);
format_names = names(format_data)[format_names];
export_data <- format_data[,c("identifier...2", format_names)]
names(export_data)[names(export_data) == "identifier...2"] <- "id"

remove_idx = vector()
for (r in 1:nrow(export_data))
{
  empty = TRUE
  for (c in format_names)
  {
    if (!is.na(export_data[r,c]))
    {
      empty = FALSE
      break
    }
  }
  if (empty)
  {
    remove_idx = c(remove_idx, r)
  }
}
export_data = export_data[-remove_idx,]

write.csv(export_data, "C:/Users/beste/Documents/RUG/BachelorThesis/DataSpec/id_list_format.csv", row.names = FALSE)

table = vector()
for (name in format_names) {
  table = c(table, format_data[,name])
}
table = table(table)
table = as.data.frame(table)

remove_idx = vector()
for (r in 1:nrow(table))
{
  if (table[r,2] < 100)
  {
    remove_idx = c(remove_idx, r)
  }
}
table = table[-remove_idx,]

dict_file <- file("dict_file.txt")
to_write = "{"
for (r in 1:nrow(table))
{
  to_write = c(to_write, paste("\"", as.character(table[r,1]), "\" : ", as.character(r-1), ",", sep="", collapse = NULL))
}
to_write = c(to_write, "}")
writeLines(to_write, dict_file)
close(dict_file)