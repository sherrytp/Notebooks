from collections import namedtuple

MAX_CRAVINGS = 2

Item = namedtuple('Item', 'product price craving')

class DuplicateProduct(Exception):
    pass

class MaxCravingsReached(Exception): 
    pass

class Groceries: 

    def __init__(self, items=None): 
        self._items = items if items is not None else []

    def show(self): 
        for item in self._items: 
            product = f'{item.product}'
            if item.craving: 
                product += ' (craving)'
            print(f'{product:<30} | {item.price:>3}')
        print('-' * 36)
        print(f'{"Total":<30} | {self.due:>3}')

    def add(self, new_item): 
        if any(item for item in self if item.product == new_item.product): 
            raise DuplicateProduct(f'{new_item.product} already in items')
        if new_item.craving and self.num_cravings_reached: 
            raise MaxCravingsReached(f'{MAX_CRAVINGS} allowed')
        self._items.append(new_item)

    def delete(self, product): 
        for i, item in enumerate(self): 
            if item.product == product: 
                self._items.pop(i)
                break
        else: 
            raise IndexError(f'{product} not in cart')

    def search(self, search): 
        for item in self: 
            if search.lower() in item.product: 
                yield item

    @property
    def due(self): 
        return sum(item.price for item in self)

    @property
    def num_cravings_reached(self): 
        return len([item for item in self if item.craving]) >= MAX_CRAVINGS

    def __len__(self): 
        return len(self._items)

    def __getitem__(self, index): 
        return self._items[index]

