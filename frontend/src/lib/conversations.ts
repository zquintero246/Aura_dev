import {
  listConversations as listFromChat,
  startConversation as startOnChat,
  deleteConversation as removeOnChat,
  updateConversation as updateOnChat,
  Conversation,
} from './chatApi';

export type { Conversation };

export async function listConversations(): Promise<Conversation[]> {
  return await listFromChat();
}

export async function createConversation(
  title?: string,
  options?: { participants?: string[] },
): Promise<Conversation> {
  return await startOnChat(title, options);
}

export async function deleteConversation(id: string): Promise<void> {
  if (!id || id.startsWith('tmp-')) return; // nothing to delete remotely
  await removeOnChat(id);
}

export async function updateConversation(
  id: string,
  updates: {
    title?: string;
    bubbleColor?: string;
    backgroundColor?: string;
  },
): Promise<Conversation> {
  return await updateOnChat(id, updates);
}

export async function updateConversationTitle(id: string, title: string): Promise<Conversation> {
  return await updateConversation(id, { title });
}
