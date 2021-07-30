# def create_train_data():
#     input = open("Raw_it_data.txt", encoding = "utf8")
#     output = open("it.txt", 'a', encoding = "utf8")
#     no_of_lines = current_line = 0
#     for each in input:
#         if each != "\n" and (current_line % 70) == 0:
#             output.write("it|" + each )
#             no_of_lines += 1
#         current_line += 1
#         if no_of_lines == 2500:
#             break
#     input.close()
#     output.close()

def create_train_data():
    input = open("it.txt", encoding = "utf8")
    output = open("en.txt", 'a', encoding = "utf8")
    for each in input:
        output.write(each)
    input.close()
    output.close()

# def create_train_data():
#     input = open("test.dat", encoding = "utf8")
#     output = open("it.txt", 'a', encoding = "utf8")
#     for each in input:
#         if each.split('|')[0] == 'en':
#             output.write("en|" + each.split('|')[1] )
#     input.close()
#     output.close()

create_train_data()