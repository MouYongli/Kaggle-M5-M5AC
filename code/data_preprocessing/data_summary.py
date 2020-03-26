# # Number of events is 30, that are classified into 5 event types.
event_name = ['SuperBowl', 'ValentinesDay', 'PresidentsDay', 'LentStart', 'LentWeek2',
 'StPatricksDay', 'Purim End', 'OrthodoxEaster', 'Pesach End', 'Cinco De Mayo',
 "Mother's day", 'MemorialDay', 'NBAFinalsStart', 'NBAFinalsEnd',
 "Father's day", 'IndependenceDay', 'Ramadan starts', 'Eid al-Fitr',
 'LaborDay', 'ColumbusDay', 'Halloween', 'EidAlAdha', 'VeteransDay',
 'Thanksgiving', 'Christmas', 'Chanukah End', 'NewYear', 'OrthodoxChristmas',
 'MartinLutherKingDay', 'Easter']
event_type = ['Sporting', 'Cultural', 'National', 'Religious']

# # Number of states are 3 and in total 10 stores are opened in those states
state = ['CA', 'TX', 'WI']
store = ['CA_1', 'CA_2', 'CA_3',  'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']

category = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
department = ['FOODS_1', 'FOODS_2', 'FOODS_3', 'HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2']

# # Number of items under the categories
# # Total number of items is 3049 combined with store id (10 stores) yields in total 30490 to predict
# # FOODS_1 216
# # FOODS_2 398
# # FOODS_3 823
# # HOBBIES_1 416
# # HOBBIES_2 149
# # HOUSEHOLD_1 532
# # HOUSEHOLD_2 515
# sales_train_validation = pd.read_csv('./sales_train_validation.csv')
# item_id = sales_train_validation['item_id'].unique()
# import collections
# files = collections.defaultdict(list)
# for id in item_id:
#     files[id[:-4]].append(id[-3:])
# department = ['FOODS_1', 'FOODS_2', 'FOODS_3', 'HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2']
# for dept in department:
#     print(dept, len(files[dept]))
