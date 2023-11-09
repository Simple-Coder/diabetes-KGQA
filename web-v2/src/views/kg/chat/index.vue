<template>
  <vue-advanced-chat
    :current-user-id="currentUserId"
    :rooms="JSON.stringify(rooms)"
    :messages="JSON.stringify(messages)"
    :room-actions="JSON.stringify(roomActions)"
    :rooms-loaded="true"
    :messages-loaded="true"
    @fetch-messages="fetchMessages"
    @send-message="sendMessage"
  />
</template>

<script>
import { register } from 'vue-advanced-chat'

register()

// Or if you used CDN import
// window['vue-advanced-chat'].register()

export default {
  data() {
    return {
      currentUserId: '1234',
      rooms: [
        {
          roomId: '1',
          roomName: 'Room 1',
          avatar: 'assets/imgs/people.png',
          unreadCount: 4,
          index: 3,
          lastMessage: {
            _id: 'xyz',
            content: 'Last message received',
            senderId: '1234',
            username: 'John Doe',
            timestamp: '10:20',
            saved: true,
            distributed: false,
            seen: false,
            new: true
          },
          users: [
            {
              _id: '1234',
              username: 'John Doe',
              avatar: 'assets/imgs/doe.png',
              status: {
                state: 'online',
                lastChanged: 'today, 14:30'
              }
            },
            {
              _id: '4321',
              username: 'John Snow',
              avatar: 'assets/imgs/snow.png',
              status: {
                state: 'offline',
                lastChanged: '14 July, 20:00'
              }
            }
          ],
          typingUsers: [4321]
        }
      ],
      messages: [],
      roomActions: [
        { name: 'inviteUser', title: 'Invite User' },
        { name: 'removeUser', title: 'Remove User' },
        { name: 'deleteRoom', title: 'Delete Room' }
      ]
    }
  },
  methods: {
    fetchMessages({ room, options }) {
      this.messagesLoaded = false
      // use timeout to imitate async server fetched data
      setInterval(() => {
        this.messages.push({
          _id: 7890,
          indexId: 12092,
          content: 'Message 1',
          senderId: '1234',
          username: 'John Doe',
          avatar: 'assets/imgs/doe.png',
          date: '13 November',
          timestamp: '10:20',
          system: false,
          saved: true,
          distributed: true,
          seen: true,
          deleted: false,
          // failure: true,
          disableActions: false,
          disableReactions: false,
          files: [
            /* {
              name: 'My File',
              size: 67351,
              type: 'png',
              audio: true,
              duration: 14.4,
              url: 'https://firebasestorage.googleapis.com/...',
              preview: 'data:image/png;base64,iVBORw0KGgoAA...',
              progress: 88
            }*/
          ]
          /* reactions: {
            AAA: [
              '1234', // USER_ID
              '4321'
            ],
            BBB: [
              '1234'
            ]
          },*/
          /* replyMessage: {
             content: 'Reply Message',
             senderId: '4321',
             files: [
               {
                 name: 'My Replied File',
                 size: 67351,
                 type: 'png',
                 audio: true,
                 duration: 14.4,
                 url: 'https://firebasestorage.googleapis.com/...',
                 preview: 'data:image/png;base64,iVBORw0KGgoAA...'
               }
             ]
           },*/
        })
        this.messagesLoaded = true
      }, 5000)

      setInterval(() => {
        this.messages.push({
          _id: 7890,
          indexId: 12092,
          content: 'Message 1',
          senderId: '4321',
          username: 'John Doe',
          avatar: 'assets/imgs/doe.png',
          date: '13 November',
          timestamp: '10:20',
          system: false,
          saved: true,
          distributed: true,
          seen: true,
          deleted: false,
          // failure: true,
          disableActions: false,
          disableReactions: false,
          files: [
            /* {
              name: 'My File',
              size: 67351,
              type: 'png',
              audio: true,
              duration: 14.4,
              url: 'https://firebasestorage.googleapis.com/...',
              preview: 'data:image/png;base64,iVBORw0KGgoAA...',
              progress: 88
            }*/
          ]
          /* reactions: {
            AAA: [
              '1234', // USER_ID
              '4321'
            ],
            BBB: [
              '1234'
            ]
          },*/
          /* replyMessage: {
             content: 'Reply Message',
             senderId: '4321',
             files: [
               {
                 name: 'My Replied File',
                 size: 67351,
                 type: 'png',
                 audio: true,
                 duration: 14.4,
                 url: 'https://firebasestorage.googleapis.com/...',
                 preview: 'data:image/png;base64,iVBORw0KGgoAA...'
               }
             ]
           },*/
        })
        this.messagesLoaded = true
      }, 5000)
    },
    sendMessage({ roomId, content, files, replyMessage, usersTag }) {
      console.log(content)
      this.messages.push({
        _id: roomId,
        indexId: 12092,
        content: content,
        senderId: '1234',
        username: 'John Doe',
        avatar: 'assets/imgs/doe.png',
        date: '13 November',
        timestamp: '10:20',
        system: false,
        saved: true,
        distributed: true,
        seen: true,
        deleted: false,
        // failure: true,
        disableActions: false,
        disableReactions: false,
        files: [
          /* {
            name: 'My File',
            size: 67351,
            type: 'png',
            audio: true,
            duration: 14.4,
            url: 'https://firebasestorage.googleapis.com/...',
            preview: 'data:image/png;base64,iVBORw0KGgoAA...',
            progress: 88
          }*/
        ]
        /* reactions: {
          AAA: [
            '1234', // USER_ID
            '4321'
          ],
          BBB: [
            '1234'
          ]
        },*/
        /* replyMessage: {
           content: 'Reply Message',
           senderId: '4321',
           files: [
             {
               name: 'My Replied File',
               size: 67351,
               type: 'png',
               audio: true,
               duration: 14.4,
               url: 'https://firebasestorage.googleapis.com/...',
               preview: 'data:image/png;base64,iVBORw0KGgoAA...'
             }
           ]
         },*/
      })
    }
  }
}
</script>
