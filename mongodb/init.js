db.createCollection("conversations");
db.createCollection("messages");
db.createCollection("uploads");

db.conversations.insertOne({
  user_id: 1,
  title: "Primera conversación de prueba",
  created_at: new Date(),
  messages: []
});