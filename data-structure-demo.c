#include <stdio.h>
#include <stdlib.h>

typedef struct node
{
    int data;
    struct node *next;
} node;

void initNode(node *n)
{
    n->data = 1;
    n->next = NULL;
}

void insertNode(node *prev, node *current, int data)
{
    current->data = data;
    current->next = prev->next;
    prev->next = current;
}
void throughNode(node *n)
{
    for(node *temp = n; !temp->next;)
    {
        printf("%d", temp->data);
        temp = temp->next;
    }
}
int main()
{
    node *head = (node*) malloc(sizeof(node));
    node *next = (node*) malloc(sizeof(node));
    node *nnext = (node*) malloc(sizeof(node));
    initNode(head);
    initNode(next);
    initNode(nnext);
    int data = 10;
    int data2 = 100;
    insertNode(head, next, data);
    insertNode(next, nnext, data2);
    throughNode(head);
    free(head);
    free(next);
    free(nnext);
    return 0;
}