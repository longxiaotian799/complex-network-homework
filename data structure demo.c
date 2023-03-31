#include <stdio.h>
#include <stdlib.h>

typedef struct node
{
    int data;
    struct node *next;
} node;

int main()
{
    node *head = NULL;
    int data;
    head = (node*) malloc(sizeof(node));
    head->next = NULL;
    head->data = 1;
    printf("%d\n", head->data);
    
    node *n = (node*) malloc(sizeof(node));
    head->next = n;
    n->next = NULL;
    n->data = 1;
    printf("%d", n->data);
}