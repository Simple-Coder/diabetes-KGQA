const tokens = {
  admin: {
    token: 'admin-token'
  },
  editor: {
    token: 'editor-token'
  },
  tom: {
    token: 'tom-token'
  },
  jerry: {
    token: 'jerry-token'
  },
  robot: {
    token: 'robot-token'
  }
}

const users = {
  'admin-token': {
    _id: 'admin',
    roles: ['admin'],
    introduction: 'I am a super administrator',
    avatar: 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
    name: 'Super Admin',
    pwd: '123456'
  },
  'editor-token': {
    _id: 'editor',
    roles: ['editor'],
    introduction: 'I am an editor',
    avatar: 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
    name: 'Normal Editor',
    pwd: '123456'
  },
  'tom': {
    _id: 'tom',
    username: 'tom',
    roles: ['editor'],
    name: 'tom',
    avatar: 'https://66.media.tumblr.com/avatar_c6a8eae4303e_512.pnj',
    pwd: '123456'
  },
  'jerry': {
    _id: 'jerry',
    username: 'jerry',
    roles: ['editor'],
    name: 'jerry',
    introduction: 'I am Leia',
    avatar: 'https://avatarfiles.alphacoders.com/184/thumb-184913.jpg',
    pwd: '123456'
  },
  'robot': {
    _id: 'robot',
    username: 'robot',
    name: 'robot',
    roles: ['editor'],
    introduction: 'I am Robot',
    avatar:
      'https://vignette.wikia.nocookie.net/teamavatarone/images/4/45/Yoda.jpg/revision/latest?cb=20130224160049',
    pwd: '123456'
  }
}

module.exports = [
  // user login
  {
    url: '/vue-element-admin/user/login',
    type: 'post',
    response: config => {
      console.log(config)
      const {username} = config.body
      console.log(username)
      const token = tokens[username]

      // mock error
      if (!token) {
        return {
          code: 60204,
          message: 'Account and password are incorrect.'
        }
      }

      return {
        code: 20000,
        data: token
      }
    }
  },

  // get user info
  {
    url: '/vue-element-admin/user/info\.*',
    type: 'get',
    response: config => {
      const {token} = config.query
      const info = users[token]

      // mock error
      if (!info) {
        return {
          code: 50008,
          message: 'Login failed, unable to get user details.'
        }
      }

      return {
        code: 20000,
        data: info
      }
    }
  },

  // get all users
  {
    url: '/vue-element-admin/user/all',
    type: 'get',
    response: config => {
      console.log(config.query)
      // Convert the 'users' object into an array of user objects
      const allUsers = Object.keys(users).map(token => users[token])

      return {
        code: 20000,
        data: allUsers
      }
    }
  },

  // user logout
  {
    url: '/vue-element-admin/user/logout',
    type: 'post',
    response: _ => {
      return {
        code: 20000,
        data: 'success'
      }
    }
  },

  // user register
  {
    url: '/vue-element-admin/user/register',
    type: 'post',
    response: config => {
      const {username, password} = config.body;
      // const {username, password, role} = config.body;

      // Check if the username is already taken
      if (tokens[username]) {
        return {
          code: 60205,
          message: '用户名已存在！请重新输入用户名!'
        };
      }

      // Generate a new token for the registered user
      const newToken = `${username}-token`
      const role = 'editor'
      // Create a new user entry
      users[newToken] = {
        _id: username,
        roles: [role],
        introduction: `I am a ${role}`,
        avatar: 'https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif',
        name: username,
        pwd: password
      };

      // Add the new user's token to the tokens object
      tokens[username] = {
        token: newToken
      };

      return {
        code: 20000,
        data: newToken
      };
    }
  },


]
