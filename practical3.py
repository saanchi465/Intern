import numpy as np

def input_matrix(name):
    rows = int(input(f"Enter number of rows for {name}: "))
    cols = int(input(f"Enter number of columns for {name}: "))
    print(f"Enter elements of {name} row-wise:")
    
    elements = []
    for i in range(rows):
        row = list(map(float, input().split()))
        elements.append(row)
        
    return np.array(elements)

print("===== MATRIX OPERATIONS TOOL =====")

A = input_matrix("Matrix A")
B = input_matrix("Matrix B")

while True:
    print("\nChoose an operation:")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Transpose")
    print("5. Determinant")
    print("6. Exit")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        if A.shape == B.shape:
            print("\nResult of Addition:\n", A + B)
        else:
            print("Addition not possible (dimension mismatch)")

    elif choice == 2:
        if A.shape == B.shape:
            print("\nResult of Subtraction:\n", A - B)
        else:
            print("Subtraction not possible (dimension mismatch)")

    elif choice == 3:
        if A.shape[1] == B.shape[0]:
            print("\nResult of Multiplication:\n", np.dot(A, B))
        else:
            print("Multiplication not possible")

    elif choice == 4:
        print("\nTranspose of Matrix A:\n", A.T)
        print("\nTranspose of Matrix B:\n", B.T)

    elif choice == 5:
        if A.shape[0] == A.shape[1]:
            print("\nDeterminant of Matrix A:", np.linalg.det(A))
        else:
            print("Determinant not possible for Matrix A")

        if B.shape[0] == B.shape[1]:
            print("Determinant of Matrix B:", np.linalg.det(B))
        else:
            print("Determinant not possible for Matrix B")

    elif choice == 6:
        print("Exiting Matrix Operations Tool.")
        break

    else:
        print("Invalid choice. Try again.")
