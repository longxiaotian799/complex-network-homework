#include <stdio.h>
#include <stdlib.h>

#define MAX_QUEUE_SIZE 100

typedef struct TreeNode {
    int data;
    struct TreeNode *left, *right;
} TreeNode;

typedef TreeNode *QueueType;

//循环队列
QueueType queue[MAX_QUEUE_SIZE];
int front = 0;
int rear = 0;

void error(char* message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

void enqueue(QueueType item) {
    if ((rear + 1) % MAX_QUEUE_SIZE == front) {
        error("队列已满");
    }
    queue[rear] = item;
    rear = (rear + 1) % MAX_QUEUE_SIZE;
}

QueueType dequeue() {
    QueueType item;
    if (front == rear) {
        error("队列为空");
    }
    item = queue[front];
    front = (front + 1) % MAX_QUEUE_SIZE;
    return item;
}

void level_order(TreeNode *ptr) {
    if (ptr == NULL) return;
    enqueue(ptr);
    while (front != rear) {
        ptr = dequeue();
        printf("[%d] ", ptr->data);
        if (ptr->left) enqueue(ptr->left);
        if (ptr->right) enqueue(ptr->right);
    }
}

TreeNode n1 = { 5, NULL, NULL };
TreeNode n2 = { 12, &n1, NULL };
TreeNode n3 = { 3, NULL, NULL };
TreeNode n4 = { 9, NULL, NULL };
TreeNode n5 = { 18, &n3, &n4 };
TreeNode n6 = { 15, &n2, &n5 };
TreeNode *root = &n6;

int main(void) {
    printf("level order traversal = ");
    level_order(root);
    printf("\n");
    return 0;
}


