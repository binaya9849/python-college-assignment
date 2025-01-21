shopping_list = []
choice = None

while choice != "3":
    print("\nWould you like to:")
    print("(1) Add items")
    print("(2) Remove items")
    print("(3) Quit")
    choice = input("Enter your choice: ")

    if choice == "1":
        item = input("What will be added?: ")
        shopping_list.append(item)
        print(f'"{item}" has been added to the list.')

    elif choice == "2":
        if shopping_list:
            print(f"There are {len(shopping_list)} items in the list:")
            for i, item in enumerate(shopping_list):
                print(f"{i}: {item}")
            index = input("Which item number will be deleted?: ")

            if index.isdigit():
                index = int(index)
                if 0 <= index < len(shopping_list):
                    removed_item = shopping_list.pop(index)
                    print(f'"{removed_item}" has been removed from the list.')
                else:
                    print("Invalid index. Please select a valid item number.")
            else:
                print("Invalid input. Please enter a number.")
        else:
            print("The list is empty. Nothing to remove.")

    elif choice == "3":
        print("The following items remain in the list:")
        for item in shopping_list:
            print(item)
        print("Goodbye!")
    else:
        print("Invalid selection. Please enter 1, 2, or 3.")
