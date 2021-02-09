class Entity:
    def __init__(self, name):
        self.name = name
        self.relation_list = []
        
    def add_new_fact(self, fact):
        self.relation_list.append(fact)

    def show_triples(self):
        for facts in self.relation_list:
            print((self, facts[0], facts[1]))

    def __repr__(self):
        return self.name
        
class Relation:
    def __init__(self, relation_name, bi=False):
        self.name = relation_name
        
    def __repr__(self):
        return self.name
        

class SubGraph:
    def __init__(self, entity_list, relation_list):
        self.entity_list = entity_list
        self.relation_list = relation_list
        self.triple_list = []
        self.mention_list = []

    def first_build(self):
        print('graph를 build합니다...')
        for entity in self.entity_list.data():
            for facts in entity.relation_list:
                self.triple_list.append((entity, facts[0], facts[1]))

    def show_triples(self):
        print('graph 안에 존재하는 triple은...')
        for triple in self.triple_list:
            print(triple)

    def add_mention(self, mention):
        self.mention_list.append(mention)

    def check_mention(self):
        print('새롭게 입력된 mention을 체크합니다...')
        new_triple = []
        for mention in self.mention_list:
            for entity in self.entity_list.data():
                if entity.name in mention:
                    new_triple.append(entity)
        new_triple.insert(1, 'has_relations')
        new_triple = tuple(new_triple)
        self.triple_list.append(new_triple)

        
        

class EntityList:
    def __init__(self):
        self.entities = []
        
    def add_entity(self, *entity):
        for i in entity:
            self.entities.append(i)
    
    def data(self):
        return self.entities
    

class RelationList:
    def __init__(self):
        self.relations = []
        
    def add_relation(self, *relation):
        for i in relation:
            self.relations.append(i)
    
    def data(self):
        return self.relations

e1 = Entity('Player 1')
e2 = Entity('Team A')
e3 = Entity('Team B')
e4 = Entity('Player 2')
r1 = Relation('was_in', True)
r2 = Relation('is_rival_with')
r3 = Relation('moved_to', True)

entity_list = EntityList()
entity_list.add_entity(e1, e2, e3, e4)
entity_list.data()

relation_list = RelationList()
relation_list.add_relation(r1, r2, r3)
relation_list.data()

e1.add_new_fact((r1, e3))
e1.add_new_fact((r3, e2))
e2.add_new_fact((r2, e3))
e3.add_new_fact((r2, e2))
e4.add_new_fact((r1, e2))


graph = SubGraph(entity_list, relation_list)
graph.first_build()
graph.show_triples()

m1 = "But surely the biggest surprise is Player 1's drop in value, despite his impressive record of 53 goals and 14 assists in 75 appearances for Team B."
graph.add_mention(m1)

print(graph.mention_list)
graph.check_mention()
graph.show_triples()