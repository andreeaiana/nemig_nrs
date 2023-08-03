# NeMig Datasets

## Overall Introduction
The user data has been collected through online studies in Germany and the US. We used the participants' implicit feedback regarding their interest in an article to build their click history, and the explicit feedback in terms of news click behaviors to construct the impression logs. To protect user privacy, we assign each user an anonymized ID. The user data is intended for non-commercial research purposes and is available under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Dataset Format

The German and English user datasets contain two files each.

| File Name | Description | 
|------|-------------|
| behaviors.tsv    |    The click history and impression logs of users.
| demographics_politics.tsv | Demographic and political information of users.

### behaviors.tsv
The behaviors.tsv file contains the users' news click histories and the impression logs. It has 4 columns divided by the tab symbol:

* Impression ID: the ID of an impression.
* User ID: The anonymized ID of an user.
* Click History: The news click history (list of news IDs) of a user before an impression. 
* Impression Log: List of news displayed to the user in a session and the user's click behavior on them (1 for click, 0 for non-click).

An example is shown in the table below:

| Column         |                                Content                                |
|------|-------------|
| Impression ID  |                                 8197                                  |
| User ID        |                                 U3794                                 |
| Click History  | news_5680 news_1586 news_1450 news_3646 news_146 news_1013 news_2447  |
| Impression Log | news_6272-0 news_6620-0 news_1254-0 news_1176-1 news_101-0 news_133-0 |

### news.tsv
The news.tsv file contains detailed information about the news articles appearing in the behaviors.tsv file. It has 8 columns divided by the tab symbol:

* News ID: the ID of a news article.
* Title
* Abstract
* Title entities (entities extracted from the title of a news article)
* Abstract entities (entities extracted from the abstract of a news article)
* Category
* Sentiment polarization (the sentiment class of an article)
* Political leaning (the politica orintation of an article, obtained from the political leaning of its source media outlet)

An example of a news article is shown in the table below:

| Column         |                                Content                                |
|------|-------------|
| News ID  |                                 news_2177                                  |
| Title        | Biden’s next executive actions address family separations, legal immigration, and asylum |
| Abstract  | Biden is continuing to prioritize immigration during his first weeks in office, but isn’t going as far as activists hoped. |
| Title entities | [{'SurfaceForms': 'Biden', 'WikidataId': 'https://www.wikidata.org/wiki/Q6279'}]|
| Abstract entities        | [{'SurfaceForms': 'Biden', 'WikidataId': 'https://www.wikidata.org/wiki/Q6279'}] |
| Category        |                                 separated_families_children_parents                                 |
| Sentiment label        |                                 Neutral                                 |
| Political class        |                                 left                                 |

The description of the dictionary keys in the "Title entities" or "Abstract entities" column is provided in the table below:

| Column         |                                Content                                |
|------|-------------|
| WikidataId  |                                 The entity ID in Wikidata                                  |
| SurfaceForms        |                                 The raw entity names in the text                                 |
