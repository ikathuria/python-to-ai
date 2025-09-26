# TODO: add scaler, vector, matrix, tensor
# TODO: add list, tuple, set, dictionary

class Node:
    def __init__(self, data=0, next=None):
        self.val = data
        self.next = None


class SinglyLinkedinList:
    def __init__(self):
        self.head = None

    def show(self):
        curr = self.head
        while curr:
            print(curr.val, end="->")
            curr = curr.next
        print("None")

    def insert_beginning(self, data):
        newNode = Node(data)
        newNode.next = self.head
        self.head = newNode

    def insert_after(self, data, node):
        if node is None:
            return
        newNode = Node(data)
        newNode.next = node.next
        node.next = newNode

    def insert_last(self, data):
        newNode = Node(data)

        if self.head is None:
            self.head = newNode
            return

        curr = self.head
        while curr.next is not None:
            curr = curr.next

        curr.next = newNode

    def delete_beginning(self):
        if self.head:
            self.head = self.head.next
        else:
            return

    def delete_after(self, node):
        if node is None:
            return
        node.next = node.next.next

    def delete_last(self):
        curr = self.head
        while curr.next.next is not None:
            curr = curr.next
        curr.next = None


sll = SinglyLinkedinList()
sll.insert_last(1)
sll.insert_last(2)
sll.insert_last(3)
sll.insert_last(4)
sll.insert_last(5)
sll.show()
