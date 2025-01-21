prices = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]
total_sum = 0
choice = None

print("Supermarket\n===========")

while True:
    choice = input("Please select product (1-10) or 0 to Quit: ")

    if choice.isdigit():
        choice = int(choice)
        if choice == 0:
            break
        elif 1 <= choice <= 10:
            price = prices[choice - 1]
            total_sum += price
            print(f"Product: {choice}, Price: {price}")
        else:
            print("Invalid selection. Please choose a number between 1 and 10, or 0 to quit.")
    else:
        print("Invalid input. Please enter a valid number.")

print(f"\nTotal: {total_sum}")

while True:
    try:
        payment = int(input("Payment: "))
        if payment >= total_sum:
            print(f"Change: {payment - total_sum}")
            break
        else:
            print("Insufficient payment. Please enter an amount equal to or greater than the total.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")
