
%%% Facts: Dinosaurs
% dino(Name, Species, Diet, Size, Era).
dino(raptor1, velociraptor, carnivore, small, cretaceous).  % More accurate size
dino(raptor2, velociraptor, carnivore, small, cretaceous).
dino(raptor3, velociraptor, carnivore, small, cretaceous).

dino(trex1, tyrannosaurus_rex, carnivore, huge, cretaceous).
dino(trex2, tyrannosaurus_rex, carnivore, huge, cretaceous).
dino(trex3, tyrannosaurus_rex, carnivore, huge, cretaceous).

dino(trike1, triceratops, herbivore, large, cretaceous).
dino(trike2, triceratops, herbivore, large, cretaceous).
dino(trike3, triceratops, herbivore, large, cretaceous).

dino(stego1, stegosaurus, herbivore, large, jurassic).
dino(ank1, ankylosaurus, herbivore, large, cretaceous).
dino(hetero1, heterodontosaurus, herbivore, small, jurassic).
dino(ornith1, iguanodon, herbivore, large, cretaceous).
dino(pachy1, pachycephalosaurus, herbivore, medium, cretaceous).
dino(plateo1, plateosaurus, herbivore, large, triassic).
dino(diplo1, diplodocus, herbivore, huge, jurassic).
dino(titano1, argentinosaurus, herbivore, huge, cretaceous).
dino(herrera1, herrerasaurus, carnivore, medium, triassic).
dino(cerato1, ceratosaurus, carnivore, large, jurassic).
dino(allo1, allosaurus, carnivore, large, jurassic).
dino(megalo1, megalosaurus, carnivore, large, jurassic).
dino(aves1, archaeopteryx, omnivore, small, jurassic).

%%% Facts: Species taxonomy (corrected hierarchy)
species_child(stegosaurus, stegosauridae).
species_child(stegosauridae, stegosauria).
species_child(stegosauria, thyreophora).

species_child(ankylosaurus, ankylosauridae).
species_child(ankylosauridae, ankylosauria).
species_child(ankylosauria, thyreophora).

species_child(thyreophora, ornithischia).

species_child(heterodontosaurus, heterodontosauridae).
species_child(heterodontosauridae, heterodontosauria).
species_child(heterodontosauria, ornithischia).

species_child(iguanodon, iguanodontidae).
species_child(iguanodontidae, ornithopoda).
species_child(ornithopoda, cerapoda).

species_child(pachycephalosaurus, pachycephalosauridae).
species_child(pachycephalosauridae, pachycephalosauria).
species_child(pachycephalosauria, marginocephalia).

species_child(triceratops, ceratopsidae).
species_child(ceratopsidae, ceratopsia).
species_child(ceratopsia, marginocephalia).

species_child(marginocephalia, cerapoda).
species_child(cerapoda, ornithischia).

species_child(plateosaurus, plateosauridae).
species_child(plateosauridae, prosauropoda).
species_child(prosauropoda, sauropodomorpha).

species_child(diplodocus, diplodocidae).
species_child(diplodocidae, diplodocoidea).
species_child(diplodocoidea, sauropoda).

species_child(argentinosaurus, titanosauridae).
species_child(titanosauridae, titanosauria).
species_child(titanosauria, macronaria).
species_child(macronaria, sauropoda).
species_child(sauropoda, sauropodomorpha).

species_child(herrerasaurus, herrerasauridae).
species_child(herrerasauridae, herrerasauria).
species_child(herrerasauria, saurischia).

species_child(ceratosaurus, ceratosauridae).
species_child(ceratosauridae, ceratosauria).
species_child(ceratosauria, theropoda).

species_child(allosaurus, allosauridae).
species_child(allosauridae, allosauroidea).
species_child(allosauroidea, carnosauria).
species_child(carnosauria, tetanurae).

species_child(megalosaurus, megalosauridae).
species_child(megalosauridae, megalosauroidea).
species_child(megalosauroidea, tetanurae).

species_child(tyrannosaurus_rex, tyrannosauridae).
species_child(tyrannosauridae, tyrannosauroidea).
species_child(tyrannosauroidea, coelurosauria).

species_child(velociraptor, dromaeosauridae).
species_child(dromaeosauridae, deinonychosauria).
species_child(deinonychosauria, maniraptora).

species_child(archaeopteryx, archaeopterygidae).
species_child(archaeopterygidae, aves).
species_child(aves, maniraptora).

species_child(maniraptora, coelurosauria).
species_child(coelurosauria, tetanurae).
species_child(tetanurae, theropoda).
species_child(theropoda, saurischia).

species_child(sauropodomorpha, saurischia).
species_child(ornithischia, dinosauria).
species_child(saurischia, dinosauria).

%%% Facts: Enclosures and connections
enclosure(paddock_a).
enclosure(paddock_b).
enclosure(paddock_c).
enclosure(paddock_d).
enclosure(paddock_e).
enclosure(service_corridor).
enclosure(cafeteria).

% in_enclosure(DinoName, Enclosure).
in_enclosure(raptor1, paddock_a).
in_enclosure(raptor2, paddock_a).
in_enclosure(raptor3, paddock_a).

in_enclosure(trex1, paddock_b).
in_enclosure(trex2, paddock_b).
in_enclosure(trex3, paddock_b).

in_enclosure(herrera1, paddock_b).
in_enclosure(cerato1, paddock_b).
in_enclosure(allo1, paddock_b).
in_enclosure(megalo1, paddock_b).

in_enclosure(trike1, paddock_c).
in_enclosure(trike2, paddock_c).
in_enclosure(trike3, paddock_c).

in_enclosure(stego1, paddock_d).
in_enclosure(ank1, paddock_d).
in_enclosure(hetero1, paddock_d).
in_enclosure(ornith1, paddock_d).
in_enclosure(pachy1, paddock_d).

in_enclosure(plateo1, paddock_e).
in_enclosure(diplo1, paddock_e).
in_enclosure(titano1, paddock_e).
in_enclosure(aves1, paddock_e).  % Moved from paddock_a for better grouping

% connected(EnclosureA, EnclosureB) (bidirectional implied)
connected(paddock_a, service_corridor).
connected(service_corridor, paddock_b).
connected(paddock_b, paddock_c).
connected(paddock_c, paddock_d).
connected(paddock_d, paddock_e).
connected(cafeteria, service_corridor).

%%% Facts: People and events
person(alan_grant, paleontologist).
person(ellie_sattler, paleobotanist).
person(ian_malcolm, mathematician).
person(park_worker1, guard).

escaped(raptor1, '1993-06-11').
sighted(trex1, paddock_b, '1993-06-11').
attacked(raptor1, park_worker1, '1993-06-11').
%%% Species-level predator facts (who hunts whom by species)
% predator(PredatorSpecies, PreySpecies)

% Tyrannosaurus Rex (apex predator, huge)
predator(tyrannosaurus_rex, triceratops).
predator(tyrannosaurus_rex, stegosaurus).
predator(tyrannosaurus_rex, ankylosaurus).
predator(tyrannosaurus_rex, iguanodon).
predator(tyrannosaurus_rex, pachycephalosaurus).
predator(tyrannosaurus_rex, velociraptor).
predator(tyrannosaurus_rex, ceratosaurus).
predator(tyrannosaurus_rex, allosaurus).
predator(tyrannosaurus_rex, megalosaurus).
predator(tyrannosaurus_rex, herrerasaurus).

% Large Carnivores
predator(allosaurus, stegosaurus).
predator(allosaurus, triceratops).
predator(allosaurus, iguanodon).
predator(allosaurus, pachycephalosaurus).
predator(allosaurus, plateosaurus).
predator(allosaurus, heterodontosaurus).
predator(allosaurus, velociraptor).

predator(ceratosaurus, stegosaurus).
predator(ceratosaurus, iguanodon).
predator(ceratosaurus, pachycephalosaurus).
predator(ceratosaurus, plateosaurus).
predator(ceratosaurus, heterodontosaurus).
predator(ceratosaurus, velociraptor).

predator(megalosaurus, stegosaurus).
predator(megalosaurus, iguanodon).
predator(megalosaurus, pachycephalosaurus).
predator(megalosaurus, plateosaurus).
predator(megalosaurus, heterodontosaurus).
predator(megalosaurus, velociraptor).

% Medium Carnivores
predator(herrerasaurus, heterodontosaurus).
predator(herrerasaurus, pachycephalosaurus).
predator(herrerasaurus, archaeopteryx).

% Small Carnivores (pack hunters)
predator(velociraptor, heterodontosaurus).
predator(velociraptor, pachycephalosaurus).
predator(velociraptor, archaeopteryx).

% Omnivore
predator(archaeopteryx, heterodontosaurus).  % small prey only








%%% Rule: symmetric connectivity
connected_bidirectional(X,Y) :- connected(X,Y).
connected_bidirectional(X,Y) :- connected(Y,X).

%%% Recursive rule: can_move_between/2
can_move_between(A,B) :- connected_bidirectional(A,B).
can_move_between(A,B) :-
    connected_bidirectional(A, Mid),
    Mid \= B,
    can_move_between(Mid, B).

%%% Recursive rule: predator_chain/2
predator_chain(P, Q) :- predator(P, Q).
predator_chain(P, Q) :-
    predator(P, X),
    predator_chain(X, Q).

%%% Species ancestor recursion
ancestor_species(Child, Parent) :- species_child(Child, Parent).
ancestor_species(Child, Anc) :-
    species_child(Child, Mid),
    ancestor_species(Mid, Anc).



%%% Rule: dangerous/1 (updated for small carnivores)
dangerous(Dino) :-
    dino(Dino, _, carnivore, _, _).  % All carnivores are potentially dangerous

dangerous(Dino) :-
    predator_chain(Dino, _).

%%% Rule: enclosure_dinos/2
enclosure_dinos(E, D) :- in_enclosure(D, E).

%%% Rule: can_encounter/2
can_encounter(D1, D2) :-
    in_enclosure(D1, E1),
    in_enclosure(D2, E2),
    can_move_between(E1, E2),
    D1 \= D2.  % Prevent self-encounters

%%% Enhanced rules for park management
% Rule: same_era/2 - dinosaurs from same geological period
same_era(D1, D2) :-
    dino(D1, _, _, _, Era),
    dino(D2, _, _, _, Era),
    D1 \= D2.

% Rule: compatible_sizes/2 - safe size combinations
compatible_sizes(D1, D2) :-
    dino(D1, _, _, Size1, _),
    dino(D2, _, _, Size2, _),
    \+ (Size1 = huge, Size2 = small),
    \+ (Size1 = small, Size2 = huge).

% Rule: safe_cohabitation/2
safe_cohabitation(D1, D2) :-
    dino(D1, _, herbivore, _, _),
    dino(D2, _, herbivore, _, _),
    compatible_sizes(D1, D2),
    D1 \= D2.

%%% Utility: list all dangerous dinos
all_dangerous(List) :-
    setof(D, dangerous(D), List).

%%% Example: feeding_chain/2
feeding_chain(Pred, PreyList) :-
    setof(Prey, predator_chain(Pred, Prey), PreyList), !.

feeding_chain(_, []).
