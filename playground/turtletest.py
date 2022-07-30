import turtle

t = turtle.Turtle()
s = turtle.Screen()
s.bgcolor('white')
t.pencolor('green')
t.speed(1)
for i in range(10):
    t.circle(190-1, 90)
    t.lt(98)
    t.circle(190-1, 90)
    t.lt(18)
