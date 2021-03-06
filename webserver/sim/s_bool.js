"use strict";

var HQ = require( '../helpers.js' ).mqtt;

module.exports = function( conf ) {

	var self = {

		conf: conf,

		status: conf.status.initial,

		_genStatus: function( oldStatus ) {

			if( Math.random() < conf.freq ) return oldStatus
			else return !oldStatus;
		},

		run: function( emit ) {

			if( conf.mode == 'random' ) {
				self.status = self._genStatus( self.status );
			}

			if( ! conf.disabled ) emit( conf.topic + '/status', '' + HQ.toString( self.status, 'b' ) );
		}
	}
	return self;
};
