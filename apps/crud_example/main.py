import sqlite3
from employee import Employee

# Create an connect a database
conn = sqlite3.connect("employee.db")

# Create a cursor for sql
c = conn.cursor()
c.execute(
    """CREATE TABLE employees (
            first text,
            last text,
            pay integer
            )"""
)


def insert_emp(emp):
    with conn:  # will auto commit
        c.execute(
            "INSERT INTO employees VALUES (:first, :last, :pay)",
            {"first": emp.first, "last": emp.last, "pay": emp.pay},
        )


def get_emps_by_name(lastname):
    # get(), we don't need to commit
    c.execute("SELECT * FROM employees WHERE last=:last", {"last": lastname})
    return c.fetchall()


def update_pay(emp, pay):
    with conn:  # will auto commit
        c.execute(
            "UPDATE employees SET pay = :pay WHERE first=:first AND last=:last",
            {"first": emp.first, "last": emp.last, "pay": pay},
        )


def remove_emp(emp):
    with conn:  # will auto commit
        c.execute(
            "DELETE FROM employees WHERE first = :first AND last = :last",
            {"first": emp.first, "last": emp.last},
        )


# Create 2 employees
emp_1 = Employee("John", "Doe", 80000)
emp_2 = Employee("Jane", "Doe", 90000)

# Insert 2 employees
print("Insert employees")
insert_emp(emp_1)
insert_emp(emp_2)

# Get employee
print("\nGet employee names")
print(get_emps_by_name(lastname="Doe"))

# Update pay
print("\nUpdate pays")
update_pay(emp_1, 15000)
print(get_emps_by_name(lastname="Doe"))

# Remove employee
print(f"\nRemove employee")
remove_emp(emp_2)
print(get_emps_by_name(lastname="Doe"))


conn.close()
